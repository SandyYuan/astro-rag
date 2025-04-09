import os
import time
import requests
from scholarly import scholarly
import PyPDF2
from io import BytesIO
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PaperCollector:
    def __init__(self, author_name="Risa Wechsler", output_dir="papers"):
        self.author_name = author_name
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def search_author(self):
        """Search for the author's profile on Google Scholar."""
        logger.info(f"Searching for {self.author_name} on Google Scholar...")
        search_query = scholarly.search_author(self.author_name)
        return next(search_query)
    
    def get_publications(self, max_papers=50):
        """Get publications from the author."""
        author = self.search_author()
        logger.info(f"Found author: {author['name']}")
        
        # Fill in the author's publications
        author = scholarly.fill(author, sections=['publications'])
        publications = author['publications'][:max_papers]
        
        logger.info(f"Found {len(publications)} publications")
        return publications
    
    def download_paper(self, publication):
        """Try to download a paper given its publication info."""
        if 'pub_url' not in publication or not publication['pub_url']:
            logger.warning(f"No URL available for: {publication['bib']['title']}")
            return None
            
        url = publication['pub_url']
        
        try:
            # For arXiv papers, adjust URL to get PDF
            if 'arxiv.org' in url and not url.endswith('.pdf'):
                if '/abs/' in url:
                    url = url.replace('/abs/', '/pdf/') + '.pdf'
            
            logger.info(f"Downloading: {url}")
            response = requests.get(url, timeout=30)
            
            if response.status_code == 200:
                # Save the PDF
                title = publication['bib']['title']
                safe_title = "".join([c if c.isalnum() else "_" for c in title[:50]])
                filename = f"{self.output_dir}/{safe_title}.pdf"
                
                with open(filename, 'wb') as f:
                    f.write(response.content)
                logger.info(f"Saved: {filename}")
                return filename
            else:
                logger.warning(f"Failed to download: {url}, status: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error downloading {url}: {str(e)}")
            return None
    
    def collect_papers(self, max_papers=20):
        """Collect papers by the author."""
        publications = self.get_publications(max_papers)
        downloaded_files = []
        
        for i, pub in enumerate(publications):
            logger.info(f"Processing paper {i+1}/{len(publications)}")
            filename = self.download_paper(pub)
            if filename:
                downloaded_files.append(filename)
            # Be nice to the servers
            time.sleep(2)
        
        logger.info(f"Downloaded {len(downloaded_files)} papers")
        return downloaded_files

if __name__ == "__main__":
    collector = PaperCollector()
    papers = collector.collect_papers(max_papers=10)
    print(f"Downloaded {len(papers)} papers") 