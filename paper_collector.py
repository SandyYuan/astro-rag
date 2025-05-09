import os
import time
import logging
import arxiv
import re
import json

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
            "skipped_not_primary": 0,
            "already_downloaded": 0,
            "found_not_downloaded": []  # Track papers found but not downloaded
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
                f'au:"R. Wechsler"',  # Initial + last name
                f'au:"R H Wechsler"',  # Initials without dots
                f'au:"R.H. Wechsler"', # Initials with dots
                f'au:"R.H.Wechsler"',  # Initials with dots, no space
                f'au:"Wechsler, R.H."', # Last name, initials with dots
                f'au:"Wechsler, R H"',  # Last name, initials without dots
                f'au:"Wechsler, R. H."'  # Last name, initials with dots and spaces
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
                    
                    # Don't break early - collect papers from all formats
                    # This ensures we don't miss papers that might be found with different name formats
                    logger.info(f"Added {len(papers)} papers from format: {name_format}")
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
        
        # Look for variations of the author name, but ensure first initial is 'R'
        variations = [
            self.author_name.lower(),           # Full name: "risa h. wechsler"
            f"{author_last_name}, {author_first_name}",  # "wechsler, risa" 
            f"{author_last_name}, r", # "wechsler, r"
            f"r. {author_last_name}", # "r. wechsler"
            f"r.h. {author_last_name}", # "r.h. wechsler"
            f"r. h. {author_last_name}", # "r. h. wechsler"
            f"r h {author_last_name}", # "r h wechsler"
            f"r.h.{author_last_name}", # "r.h.wechsler"
            f"{author_last_name}, r.h.", # "wechsler, r.h."
            f"{author_last_name}, r h", # "wechsler, r h"
            f"{author_last_name}, r. h." # "wechsler, r. h."
        ]
        
        # Check if our target author is in any of the first five authors
        for author in first_five_authors:
            name = author.name.lower()
            
            # First check if it's a Wechsler
            if "wechsler" not in name:
                continue
                
            # Then check if it's Risa (or R) Wechsler
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
            
            # Check if paper already exists
            if os.path.exists(filename):
                logger.info(f"Paper already exists: {filename}")
                self.stats["already_downloaded"] += 1
                return filename
            
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
                # Create safe filename to check if it exists
                safe_title = "".join([c if c.isalnum() else "_" for c in paper.title[:50]])
                filename = f"{self.output_dir}/{safe_title}.pdf"
                
                if os.path.exists(filename):
                    logger.info(f"Paper already exists: {filename}")
                    self.stats["already_downloaded"] += 1
                    downloaded_files.append(filename)
                else:
                    filename = self.download_paper(paper)
                    if filename:
                        downloaded_files.append(filename)
                        self.stats["downloaded"] += 1
                    else:
                        # Track papers that were found but couldn't be downloaded
                        self.stats["found_not_downloaded"].append({
                            "title": paper.title,
                            "arxiv_id": paper.entry_id,
                            "arxiv_url": paper.pdf_url,
                            "authors": [author.name for author in paper.authors[:5]]
                        })
            else:
                self.stats["skipped_not_primary"] += 1
                logger.info(f"Skipping paper where {self.author_name} is not a primary author")
            
            # Be nice to the arXiv servers
            time.sleep(3)
        
        # Log summary statistics
        logger.info(f"Summary: Searched {self.stats['searched']} papers")
        logger.info(f"Downloaded {self.stats['downloaded']} new papers where {self.author_name} is a primary author")
        logger.info(f"Found {self.stats['already_downloaded']} papers that were already downloaded")
        logger.info(f"Skipped {self.stats['skipped_not_primary']} papers where {self.author_name} is not a primary author")
        
        # Report papers that were found but not downloaded
        if self.stats["found_not_downloaded"]:
            logger.info("\nPapers found but not downloaded:")
            for paper in self.stats["found_not_downloaded"]:
                logger.info(f"- {paper['title']} (arXiv: {paper['arxiv_id']})")
                logger.info(f"  URL: {paper['arxiv_url']}")
                logger.info(f"  Authors: {', '.join(paper['authors'])}")
        
        return downloaded_files

    def get_non_primary_papers(self, max_papers=1000, download=True):
        """Get a list of papers where the author is not among the first five authors."""
        papers = self.search_arxiv_for_author(max_papers)
        non_primary_papers = []
        
        if not papers:
            logger.warning(f"No papers found for author: {self.author_name}")
            return []
        
        # Create a separate directory for non-primary papers
        np_output_dir = os.path.join(self.output_dir, "papers-np")
        os.makedirs(np_output_dir, exist_ok=True)
        
        for paper in papers:
            # Get all authors
            authors = paper.authors
            
            # Check if our target author is in any position
            author_found = False
            author_position = None
            
            # Convert author name parts to lowercase for flexible matching
            author_parts = self.author_name.lower().split()
            author_first_name = author_parts[0].lower() if author_parts else ""
            author_last_name = author_parts[-1].lower() if author_parts else ""
            
            # Look for variations of the author name
            variations = [
                self.author_name.lower(),           # Full name: "risa h. wechsler"
                f"{author_last_name}, {author_first_name}",  # "wechsler, risa" 
                f"{author_last_name}, r", # "wechsler, r"
                f"r. {author_last_name}", # "r. wechsler"
                f"r.h. {author_last_name}", # "r.h. wechsler"
                f"r. h. {author_last_name}", # "r. h. wechsler"
                f"r h {author_last_name}", # "r h wechsler"
                f"r.h.{author_last_name}", # "r.h.wechsler"
                f"{author_last_name}, r.h.", # "wechsler, r.h."
                f"{author_last_name}, r h", # "wechsler, r h"
                f"{author_last_name}, r. h." # "wechsler, r. h."
            ]
            
            # Check all authors for our target
            for i, author in enumerate(authors):
                name = author.name.lower()
                for variation in variations:
                    if variation in name:
                        author_found = True
                        author_position = i + 1  # 1-based position
                        break
                if author_found:
                    break
            
            # If author is found but not in top 5, add to list
            if author_found and author_position > 5:
                paper_info = {
                    "title": paper.title,
                    "authors": [author.name for author in paper.authors],
                    "author_position": author_position,
                    "published": paper.published,
                    "arxiv_id": paper.entry_id,
                    "arxiv_url": paper.pdf_url,
                    "categories": paper.categories
                }
                non_primary_papers.append(paper_info)
                
                # Download the paper if requested
                if download:
                    try:
                        # Create a safe filename based on the paper title
                        safe_title = "".join([c if c.isalnum() else "_" for c in paper.title[:50]])
                        filename = f"{np_output_dir}/{safe_title}.pdf"
                        
                        # Check if paper already exists
                        if not os.path.exists(filename):
                            logger.info(f"Downloading non-primary paper: {paper.title}")
                            paper.download_pdf(filename=filename)
                            logger.info(f"Successfully downloaded to {filename}")
                            
                            # Save metadata
                            metadata_filename = f"{os.path.splitext(filename)[0]}.txt"
                            with open(metadata_filename, 'w', encoding='utf-8') as f:
                                f.write(f"Title: {paper.title}\n")
                                f.write(f"Authors: {', '.join(author.name for author in paper.authors)}\n")
                                f.write(f"Published: {paper.published}\n")
                                f.write(f"Updated: {paper.updated}\n")
                                f.write(f"DOI: {paper.doi}\n" if paper.doi else "DOI: None\n")
                                f.write(f"arXiv ID: {paper.entry_id}\n")
                                f.write(f"arXiv URL: {paper.pdf_url}\n")
                                f.write(f"Categories: {', '.join(paper.categories)}\n")
                                f.write(f"Author Position: {author_position}\n")
                                f.write("\nAbstract:\n")
                                f.write(paper.summary)
                    except Exception as e:
                        logger.error(f"Error downloading paper {paper.title}: {str(e)}")
        
        # Save the list to a file
        output_file = os.path.join(np_output_dir, "non_primary_papers.json")
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(non_primary_papers, f, indent=2, default=str)
            logger.info(f"Saved {len(non_primary_papers)} non-primary papers to {output_file}")
        except Exception as e:
            logger.error(f"Error saving non-primary papers list: {str(e)}")
        
        return non_primary_papers

if __name__ == "__main__":
    collector = PaperCollector()
    
    # # Get primary author papers
    # papers = collector.collect_papers(max_papers=500)
    # print(f"Downloaded {len(papers)} primary author papers")
    
    # Get non-primary author papers
    non_primary = collector.get_non_primary_papers(max_papers=1000, download=True)
    print(f"\nFound {len(non_primary)} papers where {collector.author_name} is not a primary author")
    print("\nPaper titles:")
    for i, paper in enumerate(non_primary, 1):
        print(f"{i}. {paper['title']}")
        print(f"   Position: {paper['author_position']}")
        print(f"   arXiv: {paper['arxiv_id']}")
        print() 