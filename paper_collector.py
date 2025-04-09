import os
import time
import logging
import arxiv
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PaperCollector:
    def __init__(self, author_name="Risa H. Wechsler", output_dir="papers"):
        self.author_name = author_name
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize stats
        self.stats = {
            "searched": 0,
            "downloaded": 0,
            "skipped_not_primary": 0
        }
    
    def search_arxiv_for_author(self, max_results=100):
        """Search arXiv for papers by the author."""
        # Format author name for search - try different formats
        name_parts = self.author_name.split()
        
        # Build an effective list of search formats based on test results
        name_formats = []
        
        # Default to most effective formats first
        name_formats.append(f'au:"{self.author_name}"')  # Full name in quotes
        
        if len(name_parts) >= 2:
            first_name = name_parts[0]
            last_name = name_parts[-1]
            
            # Add variations we found that work well
            name_formats.extend([
                f'au:Wechsler_R',  # Format found to be very effective in our tests
                f'au:Wechsler',     # Simple last name search
                f'au:"Wechsler, R"', # Last name, first initial format
                f'au:"R. H. Wechsler"', # Initial format
                f'au:"R. Wechsler"'  # Initial + last name
            ])
            
            # Additional formats with provided name
            if first_name != "Risa" or last_name != "Wechsler":
                name_formats.extend([
                    f'au:{last_name}_{first_name[0]}',  # LastName_FirstInitial
                    f'au:"{last_name}, {first_name[0]}"',  # "LastName, FirstInitial"
                    f'au:{last_name}'  # Just LastName
                ])
        
        logger.info(f"Will try the following search formats: {name_formats}")
        
        all_papers = []
        
        # Try each name format until we find papers
        for name_format in name_formats:
            logger.info(f"Searching arXiv using format: {name_format}")
            
            try:
                search = arxiv.Search(
                    query=name_format,
                    max_results=max_results,
                    sort_by=arxiv.SortCriterion.SubmittedDate,
                    sort_order=arxiv.SortOrder.Descending
                )
                
                papers = list(search.results())
                logger.info(f"Found {len(papers)} papers with search format: {name_format}")
                
                if papers:
                    all_papers.extend(papers)
                    
                    # Only break if we found a significant number of papers,
                    # otherwise continue trying other formats and combine results
                    if len(papers) > 10:
                        break
            except Exception as e:
                logger.error(f"Error searching with format {name_format}: {str(e)}")
                continue
        
        # Deduplicate papers by ID
        unique_papers = []
        seen_ids = set()
        for paper in all_papers:
            paper_id = paper.entry_id
            if paper_id not in seen_ids:
                seen_ids.add(paper_id)
                unique_papers.append(paper)
        
        logger.info(f"Found {len(unique_papers)} unique papers on arXiv by {self.author_name}")
        self.stats["searched"] = len(unique_papers)
        
        # Debug: print first few authors of each paper
        if unique_papers:
            logger.info("First paper authors samples:")
            for i, paper in enumerate(unique_papers[:min(3, len(unique_papers))]):
                authors_str = ", ".join(author.name for author in paper.authors[:5])
                logger.info(f"Paper {i+1}: {authors_str} - Title: {paper.title}")
        
        return unique_papers
    
    def is_primary_author(self, paper):
        """Check if the author is among the first five authors of the paper."""
        # Get all authors
        authors = paper.authors
        
        # Check only first five authors (or fewer if there are less than five)
        first_five_authors = authors[:min(5, len(authors))]
        
        # Debug: print the authors we're checking
        author_names = [author.name for author in first_five_authors]
        logger.info(f"Checking author position among: {', '.join(author_names)}")
        
        # Convert author name parts to lowercase for flexible matching
        author_parts = self.author_name.lower().split()
        author_first_name = author_parts[0].lower() if author_parts else ""
        author_last_name = author_parts[-1].lower() if author_parts else ""
        
        # Look for variations of the author name
        variations = [
            self.author_name.lower(),           # Full name: "risa h. wechsler"
            f"{author_last_name}, {author_first_name}",  # "wechsler, risa" 
            f"{author_last_name}, {author_first_name[0]}", # "wechsler, r"
            f"{author_first_name[0]}. {author_last_name}", # "r. wechsler"
            f"{author_first_name[0]}.h. {author_last_name}", # "r.h. wechsler"
            f"r. h. {author_last_name}",        # "r. h. wechsler"
            author_last_name                    # Just "wechsler"
        ]
        
        # Check if our target author is in any of the first five authors
        for author in first_five_authors:
            name = author.name.lower()
            
            # Try to match against our variations
            for variation in variations:
                if variation in name:
                    logger.info(f"Author matched with variation '{variation}' in '{name}'")
                    return True
        
        logger.info(f"Author {self.author_name} is not among the first five authors")
        return False
    
    def download_paper(self, paper):
        """Download a paper from arXiv."""
        try:
            # Create a safe filename based on the paper title
            title = paper.title
            safe_title = "".join([c if c.isalnum() else "_" for c in title[:50]])
            filename = f"{self.output_dir}/{safe_title}.pdf"
            
            # Download the paper
            logger.info(f"Downloading paper: {title}")
            paper.download_pdf(filename=filename)
            logger.info(f"Successfully downloaded to {filename}")
            
            # Record metadata about the paper
            self._save_paper_metadata(paper, filename)
            
            return filename
        except Exception as e:
            logger.error(f"Error downloading paper: {str(e)}")
            return None
    
    def _save_paper_metadata(self, paper, pdf_filename):
        """Save metadata about the downloaded paper."""
        base_filename = os.path.splitext(pdf_filename)[0]
        metadata_filename = f"{base_filename}.txt"
        
        try:
            with open(metadata_filename, 'w', encoding='utf-8') as f:
                f.write(f"Title: {paper.title}\n")
                f.write(f"Authors: {', '.join(author.name for author in paper.authors)}\n")
                f.write(f"Published: {paper.published}\n")
                f.write(f"Updated: {paper.updated}\n")
                f.write(f"DOI: {paper.doi}\n" if paper.doi else "DOI: None\n")
                f.write(f"arXiv ID: {paper.entry_id}\n")
                f.write(f"arXiv URL: {paper.pdf_url}\n")
                f.write(f"Categories: {', '.join(paper.categories)}\n")
                f.write("\nAbstract:\n")
                f.write(paper.summary)
            logger.info(f"Saved metadata to {metadata_filename}")
        except Exception as e:
            logger.error(f"Error saving metadata: {str(e)}")
    
    def collect_papers(self, max_papers=100):
        """Collect papers by the author from arXiv."""
        papers = self.search_arxiv_for_author(max_papers)
        downloaded_files = []
        
        if not papers:
            logger.warning(f"No papers found for author: {self.author_name}")
            logger.info("Try modifying the author name format or checking if the author has papers on arXiv")
            return []
        
        for i, paper in enumerate(papers):
            logger.info(f"Processing paper {i+1}/{len(papers)}: {paper.title}")
            
            # Check if the author is among the first five authors
            if self.is_primary_author(paper):
                filename = self.download_paper(paper)
                if filename:
                    downloaded_files.append(filename)
                    self.stats["downloaded"] += 1
            else:
                self.stats["skipped_not_primary"] += 1
                logger.info(f"Skipping paper where {self.author_name} is not a primary author")
            
            # Be nice to the arXiv servers
            time.sleep(3)
        
        # Log summary statistics
        logger.info(f"Summary: Searched {self.stats['searched']} papers")
        logger.info(f"Downloaded {self.stats['downloaded']} papers where {self.author_name} is a primary author")
        logger.info(f"Skipped {self.stats['skipped_not_primary']} papers where {self.author_name} is not a primary author")
        
        return downloaded_files

if __name__ == "__main__":
    collector = PaperCollector()
    papers = collector.collect_papers(max_papers=500)
    print(f"Downloaded {len(papers)} papers") 