import requests
from bs4 import BeautifulSoup
import os
import time
import logging
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from urllib.parse import urljoin, urlparse
import hashlib
from typing import List, Dict, Set
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

class WebCrawler:
    """Advanced web crawler for collecting face images from public websites"""
    
    def __init__(self):
        """Initialize the web crawler"""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Setup directories
        self.crawl_folder = 'static/crawled_images'
        self.thumbnails_folder = 'static/thumbnails'
        os.makedirs(self.crawl_folder, exist_ok=True)
        
        # Tracking
        self.downloaded_urls = set()
        self.visited_pages = set()
        self.crawl_stats = {
            'pages_visited': 0,
            'images_downloaded': 0,
            'faces_detected': 0,
            'errors': 0
        }
        
        # Configure selenium
        self.setup_selenium()
        
        logging.info("WebCrawler initialized successfully")
    
    def setup_selenium(self):
        """Setup selenium webdriver for dynamic content"""
        try:
            chrome_options = Options()
            chrome_options.add_argument('--headless')
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument('--window-size=1920,1080')
            chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
            
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            logging.info("Selenium webdriver initialized")
            
        except Exception as e:
            logging.warning(f"Selenium setup failed: {e}. Will use requests only.")
            self.driver = None
    
    def get_target_websites(self) -> List[Dict]:
        """Get list of target websites for crawling"""
        return [
            {
                'name': 'Unsplash People',
                'base_url': 'https://unsplash.com/s/photos/people',
                'type': 'unsplash',
                'max_pages': 5
            },
            {
                'name': 'Pexels People',
                'base_url': 'https://www.pexels.com/search/people/',
                'type': 'pexels',
                'max_pages': 5
            },
            {
                'name': 'Pixabay People',
                'base_url': 'https://pixabay.com/photos/search/people/',
                'type': 'pixabay',
                'max_pages': 3
            }
        ]
    
    def crawl_websites(self, max_images: int = 100) -> Dict:
        """
        Crawl multiple websites for face images
        
        Args:
            max_images: Maximum number of images to download
            
        Returns:
            Dictionary with crawling statistics
        """
        target_websites = self.get_target_websites()
        images_collected = 0
        
        for website in target_websites:
            if images_collected >= max_images:
                break
                
            logging.info(f"Starting crawl of {website['name']}")
            
            try:
                if website['type'] == 'unsplash':
                    collected = self.crawl_unsplash(website, max_images - images_collected)
                elif website['type'] == 'pexels':
                    collected = self.crawl_pexels(website, max_images - images_collected)
                elif website['type'] == 'pixabay':
                    collected = self.crawl_pixabay(website, max_images - images_collected)
                else:
                    collected = self.crawl_generic(website, max_images - images_collected)
                
                images_collected += collected
                logging.info(f"Collected {collected} images from {website['name']}")
                
                # Respectful delay between websites
                time.sleep(2)
                
            except Exception as e:
                logging.error(f"Error crawling {website['name']}: {e}")
                self.crawl_stats['errors'] += 1
                continue
        
        if self.driver:
            self.driver.quit()
        
        return self.crawl_stats
    
    def crawl_unsplash(self, website: Dict, max_images: int) -> int:
        """Crawl Unsplash for people photos"""
        collected = 0
        
        try:
            if not self.driver:
                return 0
            
            self.driver.get(website['base_url'])
            time.sleep(3)
            
            # Scroll to load more images
            for scroll in range(3):
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(2)
            
            # Find image elements
            images = self.driver.find_elements(By.CSS_SELECTOR, 'img[srcset]')
            
            for img in images[:max_images]:
                if collected >= max_images:
                    break
                    
                try:
                    # Get high-res image URL
                    srcset = img.get_attribute('srcset')
                    if srcset:
                        urls = [url.strip().split(' ')[0] for url in srcset.split(',')]
                        img_url = urls[-1]  # Get highest resolution
                        
                        if self.download_image(img_url, 'unsplash'):
                            collected += 1
                            
                        time.sleep(1)  # Rate limiting
                        
                except Exception as e:
                    logging.warning(f"Error processing Unsplash image: {e}")
                    continue
            
        except Exception as e:
            logging.error(f"Error in Unsplash crawling: {e}")
        
        return collected
    
    def crawl_pexels(self, website: Dict, max_images: int) -> int:
        """Crawl Pexels for people photos"""
        collected = 0
        
        try:
            if not self.driver:
                return 0
            
            self.driver.get(website['base_url'])
            time.sleep(3)
            
            # Handle cookie consent if present
            try:
                accept_btn = WebDriverWait(self.driver, 5).until(
                    EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Accept')]"))
                )
                accept_btn.click()
                time.sleep(1)
            except:
                pass
            
            # Find image elements
            images = self.driver.find_elements(By.CSS_SELECTOR, 'img.photo-item__img')
            
            for img in images[:max_images]:
                if collected >= max_images:
                    break
                    
                try:
                    img_url = img.get_attribute('src')
                    if img_url and self.download_image(img_url, 'pexels'):
                        collected += 1
                        
                    time.sleep(1)
                    
                except Exception as e:
                    logging.warning(f"Error processing Pexels image: {e}")
                    continue
            
        except Exception as e:
            logging.error(f"Error in Pexels crawling: {e}")
        
        return collected
    
    def crawl_pixabay(self, website: Dict, max_images: int) -> int:
        """Crawl Pixabay for people photos"""
        collected = 0
        
        try:
            response = self.session.get(website['base_url'])
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find image elements
            img_elements = soup.find_all('img', {'data-lazy': True})
            
            for img in img_elements[:max_images]:
                if collected >= max_images:
                    break
                    
                try:
                    img_url = img.get('data-lazy') or img.get('src')
                    if img_url and self.download_image(img_url, 'pixabay'):
                        collected += 1
                        
                    time.sleep(1)
                    
                except Exception as e:
                    logging.warning(f"Error processing Pixabay image: {e}")
                    continue
            
        except Exception as e:
            logging.error(f"Error in Pixabay crawling: {e}")
        
        return collected
    
    def crawl_generic(self, website: Dict, max_images: int) -> int:
        """Generic crawling method for other websites"""
        collected = 0
        
        try:
            response = self.session.get(website['base_url'])
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find all image elements
            img_elements = soup.find_all('img')
            
            for img in img_elements[:max_images * 2]:  # Get more to filter later
                if collected >= max_images:
                    break
                    
                try:
                    img_url = img.get('src') or img.get('data-src')
                    if img_url:
                        # Convert relative URLs to absolute
                        img_url = urljoin(website['base_url'], img_url)
                        
                        if self.download_image(img_url, 'generic'):
                            collected += 1
                            
                        time.sleep(1)
                        
                except Exception as e:
                    logging.warning(f"Error processing generic image: {e}")
                    continue
            
        except Exception as e:
            logging.error(f"Error in generic crawling: {e}")
        
        return collected
    
    def download_image(self, url: str, source: str) -> bool:
        """
        Download an image from URL
        
        Args:
            url: Image URL
            source: Source website name
            
        Returns:
            True if download successful, False otherwise
        """
        try:
            # Skip if already downloaded
            url_hash = hashlib.md5(url.encode()).hexdigest()
            if url_hash in self.downloaded_urls:
                return False
            
            # Validate URL
            if not self.is_valid_image_url(url):
                return False
            
            # Download image
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get('content-type', '')
            if not content_type.startswith('image/'):
                return False
            
            # Generate filename
            file_extension = self.get_file_extension(content_type, url)
            filename = f"{source}_{url_hash}{file_extension}"
            filepath = os.path.join(self.crawl_folder, filename)
            
            # Save image
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            # Validate image size
            if os.path.getsize(filepath) < 1024:  # Less than 1KB
                os.remove(filepath)
                return False
            
            self.downloaded_urls.add(url_hash)
            self.crawl_stats['images_downloaded'] += 1
            
            logging.debug(f"Downloaded image: {filename}")
            return True
            
        except Exception as e:
            logging.warning(f"Failed to download {url}: {e}")
            return False
    
    def is_valid_image_url(self, url: str) -> bool:
        """Check if URL is likely to be a valid image"""
        if not url or len(url) < 10:
            return False
        
        # Skip data URLs, very small images, icons
        if url.startswith('data:'):
            return False
        
        # Skip common non-photo extensions
        skip_patterns = [
            'icon', 'logo', 'avatar', 'thumbnail', 'thumb',
            '.svg', '.gif', 'placeholder', 'spinner'
        ]
        
        url_lower = url.lower()
        for pattern in skip_patterns:
            if pattern in url_lower:
                return False
        
        return True
    
    def get_file_extension(self, content_type: str, url: str) -> str:
        """Get appropriate file extension"""
        if 'jpeg' in content_type or 'jpg' in content_type:
            return '.jpg'
        elif 'png' in content_type:
            return '.png'
        elif 'webp' in content_type:
            return '.webp'
        else:
            # Try to get extension from URL
            parsed = urlparse(url)
            path = parsed.path.lower()
            if path.endswith(('.jpg', '.jpeg', '.png', '.webp')):
                return os.path.splitext(path)[1]
            else:
                return '.jpg'  # Default
    
    def get_crawled_images(self) -> List[str]:
        """Get list of all crawled image paths"""
        images = []
        if os.path.exists(self.crawl_folder):
            for filename in os.listdir(self.crawl_folder):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                    images.append(os.path.join(self.crawl_folder, filename))
        return images
    
    def cleanup_old_images(self, days: int = 7):
        """Remove old crawled images"""
        if not os.path.exists(self.crawl_folder):
            return
        
        cutoff_time = time.time() - (days * 24 * 60 * 60)
        removed = 0
        
        for filename in os.listdir(self.crawl_folder):
            filepath = os.path.join(self.crawl_folder, filename)
            if os.path.getmtime(filepath) < cutoff_time:
                try:
                    os.remove(filepath)
                    removed += 1
                except Exception as e:
                    logging.warning(f"Failed to remove old image {filepath}: {e}")
        
        logging.info(f"Cleaned up {removed} old images")
    
    def get_stats(self) -> Dict:
        """Get crawling statistics"""
        return self.crawl_stats.copy()