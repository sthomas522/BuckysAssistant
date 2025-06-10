#!/usr/bin/env python3
"""
Earl's Auction Scraper - Fixed Version with Robust Price Parsing
Complete working scraper with all the price parsing issues resolved
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
from urllib.parse import urljoin, urlparse
import json
from datetime import datetime, timedelta
import sqlite3
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict
import logging
import os

# Set up debug logging to see what's happening with price parsing
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set to DEBUG to see detailed price parsing info
# logging.getLogger(__name__).setLevel(logging.DEBUG)

@dataclass
class AuctionItem:
    lot_number: str
    description: str
    current_price: Optional[float]
    price_text: str
    bid_count: int
    source: str
    auction_id: Optional[str] = None
    end_time: Optional[str] = None
    time_remaining: Optional[str] = None
    image_urls: List[str] = None
    auction_title: Optional[str] = None
    company_name: Optional[str] = None
    scraped_at: Optional[str] = None

    def __post_init__(self):
        if self.image_urls is None:
            self.image_urls = []
        if self.scraped_at is None:
            self.scraped_at = datetime.now().isoformat()

@dataclass
class AuctionInfo:
    company_name: str
    company_url: str
    auction_title: str
    dates: str
    location: str
    bidding_notice: str
    zip_code: Optional[str]
    end_time: Optional[str] = None
    time_remaining: Optional[str] = None
    auction_id: Optional[str] = None
    scraped_at: Optional[str] = None

    def __post_init__(self):
        if self.scraped_at is None:
            self.scraped_at = datetime.now().isoformat()

class DatabaseManager:
    def __init__(self, db_path='hibid_auctions.db'):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize the database with required tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Auctions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS auctions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    auction_id TEXT UNIQUE,
                    company_name TEXT,
                    company_url TEXT,
                    auction_title TEXT,
                    dates TEXT,
                    location TEXT,
                    bidding_notice TEXT,
                    zip_code TEXT,
                    end_time TEXT,
                    time_remaining TEXT,
                    scraped_at TEXT,
                    UNIQUE(auction_id)
                )
            ''')

            # Items table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    lot_number TEXT,
                    description TEXT,
                    current_price REAL,
                    price_text TEXT,
                    bid_count INTEGER,
                    source TEXT,
                    auction_id TEXT,
                    end_time TEXT,
                    time_remaining TEXT,
                    auction_title TEXT,
                    company_name TEXT,
                    scraped_at TEXT,
                    FOREIGN KEY (auction_id) REFERENCES auctions (auction_id)
                )
            ''')

            # Images table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS item_images (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    item_id INTEGER,
                    image_url TEXT,
                    image_order INTEGER,
                    FOREIGN KEY (item_id) REFERENCES items (id)
                )
            ''')

            conn.commit()

    def save_auction(self, auction: AuctionInfo) -> int:
        """Save auction info to database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO auctions
                (auction_id, company_name, company_url, auction_title, dates,
                 location, bidding_notice, zip_code, end_time, time_remaining, scraped_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                auction.auction_id, auction.company_name, auction.company_url,
                auction.auction_title, auction.dates, auction.location,
                auction.bidding_notice, auction.zip_code, auction.end_time,
                auction.time_remaining, auction.scraped_at
            ))
            return cursor.lastrowid

    def save_item(self, item: AuctionItem) -> int:
        """Save auction item to database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO items
                (lot_number, description, current_price, price_text, bid_count,
                 source, auction_id, end_time, time_remaining, auction_title,
                 company_name, scraped_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                item.lot_number, item.description, item.current_price,
                item.price_text, item.bid_count, item.source, item.auction_id,
                item.end_time, item.time_remaining, item.auction_title,
                item.company_name, item.scraped_at
            ))

            item_id = cursor.lastrowid

            # Save images
            for i, image_url in enumerate(item.image_urls):
                cursor.execute('''
                    INSERT INTO item_images (item_id, image_url, image_order)
                    VALUES (?, ?, ?)
                ''', (item_id, image_url, i))

            return item_id

    def get_active_auctions(self, zip_code: Optional[str] = None) -> List[Dict]:
        """Get active auctions from database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            query = '''
                SELECT * FROM auctions
                WHERE datetime(scraped_at) > datetime('now', '-24 hours')
            '''
            params = []

            if zip_code:
                query += ' AND zip_code = ?'
                params.append(zip_code)

            query += ' ORDER BY scraped_at DESC'

            cursor.execute(query, params)
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def get_items_by_auction(self, auction_id: str) -> List[Dict]:
        """Get all items for a specific auction"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT i.*, GROUP_CONCAT(img.image_url) as image_urls
                FROM items i
                LEFT JOIN item_images img ON i.id = img.item_id
                WHERE i.auction_id = ?
                GROUP BY i.id
                ORDER BY CAST(i.lot_number AS INTEGER)
            ''', (auction_id,))

            columns = [desc[0] for desc in cursor.description]
            items = []
            for row in cursor.fetchall():
                item = dict(zip(columns, row))
                if item['image_urls']:
                    item['image_urls'] = item['image_urls'].split(',')
                else:
                    item['image_urls'] = []
                items.append(item)
            return items

class EarlsAuctionScraper:
    """Earl's Auction scraper with fixed price parsing"""
    
    def __init__(self, zip_code=None, db_path='hibid_auctions.db'):
        self.base_url = "https://www.earlsauction.com"
        self.auctions_url = "https://www.earlsauction.com/auctions"
        self.zip_code = zip_code
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })

        self.db = DatabaseManager(db_path)

    def get_page_content(self, url, retries=3):
        """Fetch page content with retry logic"""
        for attempt in range(retries):
            try:
                response = self.session.get(url, timeout=15)
                response.raise_for_status()
                return response.text
            except requests.RequestException as e:
                logger.warning(f"Attempt {attempt + 1} failed for {url}: {e}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    return None

    def parse_price(self, price_text):
        """Extract numeric price from price text - FIXED VERSION"""
        if not price_text:
            return None

        logger.debug(f"Parsing price from: {repr(price_text[:200])}")
        
        # Clean up the text first - remove excessive whitespace
        cleaned_text = re.sub(r'\s+', ' ', price_text.strip())
        
        # Strategy 1: Look for very specific Earl's auction patterns
        earls_specific_patterns = [
            r'Current Bid:\s*\$(\d{1,4}(?:\.\d{2})?)',  # Current Bid: $55.00
            r'Winning Bid:\s*\$(\d{1,4}(?:\.\d{2})?)',  # Winning Bid: $55.00
            r'High Bid:\s*\$(\d{1,4}(?:\.\d{2})?)',     # High Bid: $55.00
            r'Starting Bid:\s*\$(\d{1,4}(?:\.\d{2})?)', # Starting Bid: $5.00
        ]
        
        for pattern in earls_specific_patterns:
            match = re.search(pattern, cleaned_text, re.IGNORECASE)
            if match:
                try:
                    price = float(match.group(1))
                    logger.debug(f"Found price using pattern {pattern}: ${price}")
                    if 0.01 <= price <= 50000:  # Reasonable range
                        return price
                except ValueError:
                    continue
        
        # Strategy 2: Look for standalone dollar amounts but be VERY restrictive
        standalone_patterns = [
            r'(?<!\d)\$(\d{1,4}(?:\.\d{2})?)(?!\d)',  # $55.00 not part of larger number
        ]
        
        all_found_prices = []
        for pattern in standalone_patterns:
            matches = re.finditer(pattern, cleaned_text)
            for match in matches:
                try:
                    price = float(match.group(1))
                    logger.debug(f"Found potential price: ${price}")
                    if 0.01 <= price <= 50000:
                        all_found_prices.append(price)
                except (ValueError, IndexError):
                    continue
        
        if all_found_prices:
            best_price = max(all_found_prices)
            logger.debug(f"Selected best price from {all_found_prices}: ${best_price}")
            return best_price
        
        logger.debug(f"Could not parse valid price from: {repr(price_text[:100])}")
        return None

    def parse_time_remaining(self, time_text):
        """Parse time remaining from various formats"""
        if not time_text:
            return None, None

        time_text = re.sub(r'\s+', ' ', time_text.strip())

        patterns = [
            r'(\d+)d\s*(\d+)h\s*(\d+)m',
            r'(\d+)\s*days?\s*(\d+)\s*hours?\s*(\d+)\s*min',
            r'(\d+)h\s*(\d+)m',
            r'(\d+)\s*hours?\s*(\d+)\s*min',
            r'(\d+)m',
            r'(\d+)\s*min',
            r'Ends?:?\s*(.+)',
            r'Closing:?\s*(.+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, time_text, re.IGNORECASE)
            if match:
                groups = match.groups()

                if len(groups) == 3:
                    try:
                        days, hours, minutes = map(int, groups)
                        end_time = datetime.now() + timedelta(days=days, hours=hours, minutes=minutes)
                        return end_time.isoformat(), f"{days}d {hours}h {minutes}m"
                    except ValueError:
                        continue

                elif len(groups) == 2 and any(x in time_text.lower() for x in ['h', 'hour']):
                    try:
                        hours, minutes = map(int, groups)
                        end_time = datetime.now() + timedelta(hours=hours, minutes=minutes)
                        return end_time.isoformat(), f"{hours}h {minutes}m"
                    except ValueError:
                        continue

                elif len(groups) == 1:
                    if any(x in time_text.lower() for x in ['m', 'min']):
                        try:
                            minutes = int(groups[0])
                            end_time = datetime.now() + timedelta(minutes=minutes)
                            return end_time.isoformat(), f"{minutes}m"
                        except ValueError:
                            continue

        return None, time_text

    def is_valid_image_url(self, url):
        """Check if URL is a valid image URL"""
        if not url:
            return False

        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp']
        url_lower = url.lower()

        if any(ext in url_lower for ext in image_extensions):
            return True

        image_keywords = ['image', 'img', 'photo', 'picture', 'thumb', 'gallery']
        if any(keyword in url_lower for keyword in image_keywords):
            return True

        return False

    def extract_images(self, soup, base_url):
        """Extract images from page"""
        images = []
        all_imgs = soup.find_all('img')

        for img in all_imgs:
            src = img.get('src') or img.get('data-src') or img.get('data-lazy-src')
            if not src:
                continue

            full_url = urljoin(base_url, src)

            skip_patterns = [
                'logo', 'banner', 'header', 'footer', 'icon',
                'avatar', 'profile', 'social', 'ad', 'advertisement',
                'placeholder', 'loading', 'spinner', '1x1', 'tracking'
            ]

            if any(pattern in full_url.lower() for pattern in skip_patterns):
                continue

            if self.is_valid_image_url(full_url):
                width = img.get('width')
                height = img.get('height')

                if width and height:
                    try:
                        w, h = int(width), int(height)
                        if w < 50 or h < 50:
                            continue
                    except ValueError:
                        pass

                img_classes = img.get('class', [])
                if isinstance(img_classes, str):
                    img_classes = img_classes.split()

                priority_classes = ['lot', 'item', 'product', 'auction', 'photo', 'image', 'gallery']
                has_priority = any(cls.lower() in ' '.join(img_classes).lower() for cls in priority_classes)

                if full_url not in images:
                    if has_priority:
                        images.insert(0, full_url)
                    else:
                        images.append(full_url)

        cleaned_images = []
        for img_url in images[:10]:
            parsed = urlparse(img_url)
            if parsed.scheme and parsed.netloc:
                cleaned_images.append(img_url)

        return cleaned_images[:5]

    def extract_auction_id(self, url):
        """Extract auction ID from URL"""
        patterns = [
            r'/auction/(\d+)',
            r'/catalog/(\d+)',
            r'/company/(\d+)',
            r'auction_id=(\d+)',
            r'/auctions/(\d+)',
            r'/sale/(\d+)'
        ]

        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)

        return None

    def create_auction_item_safely(self, lot_number, description, current_price, price_text, bid_count, source, auction_id, end_time, time_remaining, image_urls, auction_title, company_name):
        """Create AuctionItem with price validation"""
        
        # Validate the price before creating the item
        if current_price is not None:
            if current_price < 0.01 or current_price > 100000:
                logger.warning(f"Invalid price ${current_price} for lot {lot_number}, setting to None")
                current_price = None
                price_text = "Price unavailable"
            elif current_price > 10000:
                logger.warning(f"High price ${current_price} for lot {lot_number} - please verify")
        
        return AuctionItem(
            lot_number=lot_number,
            description=description,
            current_price=current_price,
            price_text=price_text,
            bid_count=bid_count,
            source=source,
            auction_id=auction_id,
            end_time=end_time,
            time_remaining=time_remaining,
            image_urls=image_urls,
            auction_title=auction_title,
            company_name=company_name
        )

    def scrape_earls_auctions(self):
        """Main method to scrape Earl's auction listings"""
        logger.info("Scraping Earl's Auction main page...")
        
        content = self.get_page_content(self.auctions_url)
        if not content:
            logger.warning("Could not fetch content from Earl's Auction")
            return []

        soup = BeautifulSoup(content, 'html.parser')
        auctions = []

        auction_urls = self._find_direct_auction_urls(soup)
        
        if auction_urls:
            logger.info(f"Found {len(auction_urls)} direct auction URLs")
            for url in auction_urls:
                try:
                    auction = self._create_auction_from_url(url)
                    if auction:
                        auctions.append(auction)
                except Exception as e:
                    logger.error(f"Error creating auction from URL {url}: {e}")
                    continue
        else:
            logger.info("No direct auction URLs found, trying to parse auction containers")
            auction_containers = self._find_auction_containers(soup)
            
            for container in auction_containers:
                try:
                    auction = self._parse_auction_container(container)
                    if auction:
                        auctions.append(auction)
                except Exception as e:
                    logger.error(f"Error parsing auction container: {e}")
                    continue

        logger.info(f"Found {len(auctions)} auctions on Earl's Auction")
        return auctions

    def _find_direct_auction_urls(self, soup):
        """Find direct auction URLs on the main auction page"""
        auction_urls = []
        all_links = soup.find_all('a', href=True)
        
        for link in all_links:
            href = link['href']
            if re.search(r'/auctions/\d+', href) and '/lot/' not in href:
                full_url = urljoin(self.base_url, href)
                auction_urls.append(full_url)
        
        seen = set()
        unique_urls = []
        for url in auction_urls:
            if url not in seen:
                seen.add(url)
                unique_urls.append(url)
        
        return unique_urls

    def _create_auction_from_url(self, auction_url):
        """Create an AuctionInfo object from a direct auction URL"""
        try:
            auction_id = self.extract_auction_id(auction_url)
            
            url_match = re.search(r'/auctions/\d+-(.+)', auction_url)
            if url_match:
                title_from_url = url_match.group(1).replace('-', ' ').title()
            else:
                title_from_url = "Earl's Auction"
            
            auction_details = self._fetch_auction_details(auction_url)
            
            auction = AuctionInfo(
                company_name="Earl's Auction Company",
                company_url=auction_url,
                auction_title=auction_details.get('title', title_from_url),
                dates=auction_details.get('dates', 'TBD'),
                location=auction_details.get('location', 'Indianapolis, IN'),
                bidding_notice=auction_details.get('bidding_notice', ''),
                zip_code=self.zip_code,
                end_time=auction_details.get('end_time'),
                time_remaining=auction_details.get('time_remaining'),
                auction_id=auction_id
            )
            
            return auction
            
        except Exception as e:
            logger.error(f"Error creating auction from URL {auction_url}: {e}")
            return None

    def _fetch_auction_details(self, auction_url):
        """Fetch additional details from an auction page"""
        details = {}
        
        try:
            content = self.get_page_content(auction_url)
            if not content:
                return details
            
            soup = BeautifulSoup(content, 'html.parser')
            
            title_tag = soup.find('title')
            if title_tag:
                title_text = title_tag.get_text().strip()
                title_text = re.sub(r"\s*-\s*Earl's Auction Company.*$", '', title_text)
                if len(title_text) > 5:
                    details['title'] = title_text
            
            page_text = soup.get_text()
            
            pickup_match = re.search(r'(?:pickup|pick\s*up).*?at\s+([^.]+)', page_text, re.IGNORECASE)
            if pickup_match:
                location_text = pickup_match.group(1).strip()
                if len(location_text) < 100:
                    details['location'] = location_text
            
            date_patterns = [
                r'(\w+,\s+\w+\s+\d+(?:th|st|nd|rd)?)',
                r'(\d{1,2}/\d{1,2}/\d{4})',
                r'(\w+\s+\d{1,2}(?:th|st|nd|rd)?,?\s+\d{4})'
            ]
            
            for pattern in date_patterns:
                date_match = re.search(pattern, page_text)
                if date_match:
                    details['dates'] = date_match.group(1)
                    break
            
            time_keywords = ['ends', 'ending', 'closes', 'closing']
            for keyword in time_keywords:
                if keyword in page_text.lower():
                    lines = page_text.split('\n')
                    for line in lines:
                        if keyword in line.lower():
                            end_time, time_remaining = self.parse_time_remaining(line)
                            if end_time:
                                details['end_time'] = end_time
                                details['time_remaining'] = time_remaining
                                break
                    if 'end_time' in details:
                        break
            
        except Exception as e:
            logger.error(f"Error fetching auction details from {auction_url}: {e}")
        
        return details

    def _find_auction_containers(self, soup):
        """Find auction containers on Earl's auction page"""
        containers = []
        
        all_links = soup.find_all('a', href=True)
        for link in all_links:
            href = link['href']
            if re.search(r'/auctions/\d+', href) and 'lot' not in href:
                parent = link.parent
                if parent and len(parent.get_text().strip()) > 20:
                    containers.append(parent)
        
        if not containers:
            selectors = [
                'div[class*="auction"]',
                'div[class*="listing"]', 
                'div[class*="card"]',
                'article',
                '.auction-item',
                '.listing-item'
            ]
            
            for selector in selectors:
                elements = soup.select(selector)
                if elements:
                    containers.extend(elements)

        if not containers:
            all_divs = soup.find_all('div')
            for div in all_divs:
                text = div.get_text().lower()
                has_auction_content = any(keyword in text for keyword in ['auction', 'bidding', 'ends', 'lot'])
                has_image = div.find('img') is not None
                has_substantial_content = len(text.strip()) > 50
                
                if has_auction_content and has_image and has_substantial_content:
                    containers.append(div)
        
        seen = set()
        unique_containers = []
        for container in containers:
            container_id = id(container)
            if container_id not in seen:
                seen.add(container_id)
                unique_containers.append(container)
        
        return unique_containers[:20]

    def _parse_auction_container(self, container):
        """Parse individual auction container from Earl's auction page"""
        try:
            title = self._extract_title(container)
            if not title:
                return None
                
            auction_url = self._extract_auction_url(container)
            auction_id = self.extract_auction_id(auction_url) if auction_url else None
            
            location = self._extract_location(container)
            dates = self._extract_dates(container)
            end_time, time_remaining = self._extract_time_info(container)
            
            auction = AuctionInfo(
                company_name="Earl's Auction Company",
                company_url=auction_url or self.auctions_url,
                auction_title=title,
                dates=dates,
                location=location,
                bidding_notice="",
                zip_code=self.zip_code,
                end_time=end_time,
                time_remaining=time_remaining,
                auction_id=auction_id
            )
            
            return auction
            
        except Exception as e:
            logger.error(f"Error parsing auction container: {e}")
            return None

    def _extract_title(self, container):
        """Extract auction title from container"""
        title_selectors = ['h1', 'h2', 'h3', 'h4', '.title', '.auction-title']
        
        for selector in title_selectors:
            element = container.select_one(selector)
            if element:
                text = element.get_text().strip()
                if len(text) > 5:
                    return text[:200]
        
        texts = container.find_all(text=True)
        for text in texts:
            clean_text = text.strip()
            if len(clean_text) > 10 and clean_text.upper() == clean_text and 'AUCTION' in clean_text:
                return clean_text[:200]
        
        return None

    def _extract_auction_url(self, container):
        """Extract auction URL from container"""
        link = container.find('a', href=True)
        if link:
            href = link['href']
            if '/auctions/' in href and '/lot/' not in href:
                if href.startswith('http'):
                    return href
                else:
                    return urljoin(self.base_url, href)
        
        all_links = container.find_all('a', href=True)
        for link in all_links:
            href = link['href']
            if re.search(r'/auctions/\d+', href) and '/lot/' not in href:
                if href.startswith('http'):
                    return href
                else:
                    return urljoin(self.base_url, href)
        
        return None

    def _extract_location(self, container):
        """Extract location from container"""
        text = container.get_text()
        
        address_patterns = [
            r'\d+\s+[A-Za-z\s]+(?:Road|Street|Ave|Avenue|Blvd|Boulevard|Dr|Drive|Lane|Ln|Way|Court|Ct|Circle|Cir)',
            r'[A-Za-z\s]+,\s*[A-Z]{2}\s*\d{5}'
        ]
        
        for pattern in address_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(0).strip()
        
        city_state_match = re.search(r'([A-Za-z\s]+),\s*([A-Z]{2})', text)
        if city_state_match:
            return city_state_match.group(0)
        
        return "Indianapolis, IN"

    def _extract_dates(self, container):
        """Extract auction dates from container"""
        text = container.get_text()
        
        date_patterns = [
            r'\d{1,2}/\d{1,2}/\d{4}',
            r'[A-Za-z]+\s+\d{1,2},?\s+\d{4}',
            r'\d{1,2}-\d{1,2}-\d{4}'
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(0)
        
        return "TBD"

    def _extract_time_info(self, container):
        """Extract end time and time remaining from container"""
        text = container.get_text()
        
        time_keywords = ['ends', 'ending', 'closes', 'closing', 'bidding', 'until']
        for keyword in time_keywords:
            if keyword in text.lower():
                lines = text.split('\n')
                for line in lines:
                    if keyword in line.lower():
                        return self.parse_time_remaining(line)
        
        return None, None

    def scrape_auction_details(self, auction_url):
        """Scrape individual auction details and items"""
        logger.info(f"Scraping auction details: {auction_url}")
        
        content = self.get_page_content(auction_url)
        if not content:
            return []

        soup = BeautifulSoup(content, 'html.parser')
        items = []

        auction_id = self.extract_auction_id(auction_url)
        auction_title = self._extract_page_title(soup)
        
        lot_links = self._find_lot_links(soup, auction_url)
        
        if lot_links:
            logger.info(f"Found {len(lot_links)} lot links to scrape")
            for i, lot_url in enumerate(lot_links):
                try:
                    lot_item = self._scrape_individual_lot(lot_url, auction_id, auction_title)
                    if lot_item:
                        items.append(lot_item)
                    
                    if i < len(lot_links) - 1:
                        time.sleep(1)
                        
                except Exception as e:
                    logger.error(f"Error scraping lot {lot_url}: {e}")
                    continue
        else:
            logger.info("No individual lot links found, trying to parse auction page directly")
            item_containers = self._find_item_containers(soup)
            
            for i, container in enumerate(item_containers):
                try:
                    item = self._parse_item_container(container, auction_id, auction_title, auction_url, i)
                    if item:
                        items.append(item)
                except Exception as e:
                    logger.error(f"Error parsing item container {i}: {e}")
                    continue

        logger.info(f"Found {len(items)} items in auction {auction_url}")
        return items

    def _find_lot_links(self, soup, auction_url):
        """Find individual lot links on an auction page"""
        lot_links = []
        all_links = soup.find_all('a', href=True)
        
        for link in all_links:
            href = link['href']
            if '/lot/' in href:
                full_url = urljoin(auction_url, href)
                if re.search(r'/auctions/\d+/lot/\d+', full_url):
                    lot_links.append(full_url)
        
        seen = set()
        unique_links = []
        for link in lot_links:
            if link not in seen:
                seen.add(link)
                unique_links.append(link)
        
        return unique_links[:200]
    
    def _scrape_individual_lot(self, lot_url, auction_id, auction_title):
        """Scrape an individual lot page with FIXED price parsing"""
        logger.debug(f"Scraping individual lot: {lot_url}")
        
        content = self.get_page_content(lot_url)
        if not content:
            return None
        
        soup = BeautifulSoup(content, 'html.parser')
        
        try:
            lot_match = re.search(r'/lot/(\d+)', lot_url)
            lot_number = lot_match.group(1) if lot_match else "Unknown"
            
            description = self._extract_lot_description(soup, lot_url)
            current_price, price_text = self._extract_lot_price_info(soup)
            bid_count = self._extract_lot_bid_count(soup)
            image_urls = self.extract_images(soup, lot_url)
            end_time, time_remaining = self._extract_lot_time_info(soup)
            
            item = self.create_auction_item_safely(
                lot_number=lot_number,
                description=description,
                current_price=current_price,
                price_text=price_text,
                bid_count=bid_count,
                source=lot_url,
                auction_id=auction_id,
                end_time=end_time,
                time_remaining=time_remaining,
                image_urls=image_urls,
                auction_title=auction_title,
                company_name="Earl's Auction Company"
            )
            
            logger.debug(f"Successfully parsed lot {lot_number}: {description[:50]}...")
            return item
            
        except Exception as e:
            logger.error(f"Error parsing individual lot {lot_url}: {e}")
            return None
    
    def _extract_lot_description(self, soup, lot_url):
        """Extract description from individual lot page"""
        title_tag = soup.find('title')
        if title_tag:
            title_text = title_tag.get_text().strip()
            title_text = re.sub(r"\s*-\s*Earl's Auction Company.*$", '', title_text)
            if len(title_text) > 5:
                return title_text[:300]
        
        for heading_tag in ['h1', 'h2', 'h3']:
            heading = soup.find(heading_tag)
            if heading:
                heading_text = heading.get_text().strip()
                if len(heading_text) > 5 and "Earl's Auction" not in heading_text:
                    return heading_text[:300]
        
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc and meta_desc.get('content'):
            meta_text = meta_desc['content'].strip()
            if len(meta_text) > 10:
                return meta_text[:300]
        
        url_match = re.search(r'/lot/\d+-(.+)', lot_url)
        if url_match:
            url_desc = url_match.group(1).replace('-', ' ').title()
            return url_desc[:300]
        
        return "Item description not available"
    
    def _extract_lot_price_info(self, soup):
        """Extract price information from individual lot page - FIXED VERSION"""
        logger.debug("Starting price extraction from lot page")
        
        # Strategy 1: Look in specific elements that might contain price
        price_selectors = [
            '.current-bid', '.current-price', '.bid-amount', '.winning-bid', '.high-bid',
            '[class*="bid"]', '[class*="price"]', '[id*="bid"]', '[id*="price"]'
        ]
        
        for selector in price_selectors:
            elements = soup.select(selector)
            for elem in elements:
                text = elem.get_text().strip()
                logger.debug(f"Checking element {selector}: {repr(text)}")
                if text and len(text) < 50:  # Skip very long text
                    price = self.parse_price(text)
                    if price is not None:
                        logger.info(f"Found price ${price} in element {selector}")
                        return price, f"${price:.2f}"
        
        # Strategy 2: Look for script tags with JSON price data
        scripts = soup.find_all('script')
        for script in scripts:
            if script.string:
                script_text = script.string
                json_patterns = [
                    r'"current[_-]?bid"[:\s]*(\d+(?:\.\d{2})?)',
                    r'"price"[:\s]*(\d+(?:\.\d{2})?)',
                    r'"amount"[:\s]*(\d+(?:\.\d{2})?)',
                ]
                
                for pattern in json_patterns:
                    match = re.search(pattern, script_text, re.IGNORECASE)
                    if match:
                        try:
                            price = float(match.group(1))
                            if 0.01 <= price <= 50000:
                                logger.info(f"Found price ${price} in script tag")
                                return price, f"${price:.2f}"
                        except ValueError:
                            continue
        
        # Strategy 3: Look in page text line by line to avoid concatenation
        page_text = soup.get_text()
        lines = page_text.split('\n')
        for line in lines:
            line = line.strip()
            if len(line) < 100 and ('$' in line or 'bid' in line.lower() or 'price' in line.lower()):
                logger.debug(f"Checking line: {repr(line)}")
                price = self.parse_price(line)
                if price is not None:
                    logger.info(f"Found price ${price} in page line")
                    return price, f"${price:.2f}"
        
        # Strategy 4: Look for specific text patterns in small chunks
        text_chunks = re.split(r'[.!?;]\s+', page_text)
        for chunk in text_chunks:
            if len(chunk) < 200 and ('current bid' in chunk.lower() or 'winning bid' in chunk.lower()):
                logger.debug(f"Checking chunk: {repr(chunk[:100])}")
                price = self.parse_price(chunk)
                if price is not None:
                    logger.info(f"Found price ${price} in text chunk")
                    return price, f"${price:.2f}"
        
        logger.warning("Could not find any valid price information on page")
        return None, ""
    
    def _extract_lot_bid_count(self, soup):
        """Extract bid count from individual lot page"""
        bid_selectors = ['.bid-count', '.bids', '[class*="bid"]']
        
        for selector in bid_selectors:
            bid_elem = soup.select_one(selector)
            if bid_elem:
                bid_text = bid_elem.get_text()
                bid_match = re.search(r'(\d+)', bid_text)
                if bid_match:
                    try:
                        return int(bid_match.group(1))
                    except ValueError:
                        continue
        
        all_text = soup.get_text()
        bid_patterns = [
            r'(\d+)\s*Bids?',
            r'Bids?[:\s]*(\d+)',
            r'(\d+)\s*bidders?'
        ]
        
        for pattern in bid_patterns:
            match = re.search(pattern, all_text, re.IGNORECASE)
            if match:
                try:
                    return int(match.group(1))
                except ValueError:
                    continue
        
        return 0
    
    def _extract_lot_time_info(self, soup):
        """Extract time information from individual lot page"""
        time_selectors = ['.end-time', '.closing-time', '.time-remaining', '[class*="time"]']
        
        for selector in time_selectors:
            time_elem = soup.select_one(selector)
            if time_elem:
                time_text = time_elem.get_text()
                end_time, time_remaining = self.parse_time_remaining(time_text)
                if end_time:
                    return end_time, time_remaining
        
        all_text = soup.get_text()
        time_keywords = ['ends', 'ending', 'closes', 'closing', 'time left', 'remaining']
        
        for keyword in time_keywords:
            if keyword in all_text.lower():
                lines = all_text.split('\n')
                for line in lines:
                    if keyword in line.lower():
                        end_time, time_remaining = self.parse_time_remaining(line)
                        if end_time:
                            return end_time, time_remaining
        
        return None, None

    def _extract_page_title(self, soup):
        """Extract page title"""
        title_tag = soup.find('title')
        if title_tag:
            return title_tag.get_text().strip()
        
        h1 = soup.find('h1')
        if h1:
            return h1.get_text().strip()
        
        return "Earl's Auction"

    def _find_item_containers(self, soup):
        """Find item/lot containers on auction detail page"""
        containers = []
        
        selectors = [
            'div[class*="lot"]', 'div[class*="item"]', 'tr[class*="lot"]',
            'tr[class*="item"]', '.product', '.listing'
        ]
        
        for selector in selectors:
            elements = soup.select(selector)
            if elements:
                containers.extend(elements)
        
        if not containers:
            rows = soup.find_all('tr')
            for row in rows:
                text = row.get_text().lower()
                if any(keyword in text for keyword in ['lot', 'item', 'bid', '$']):
                    containers.append(row)

        return containers[:100]

    def _parse_item_container(self, container, auction_id, auction_title, auction_url, index):
        """Parse individual item container with FIXED price parsing"""
        try:
            lot_number = self._extract_lot_number(container, index)
            description = self._extract_description(container)
            if not description or len(description) < 5:
                return None
            
            current_price, price_text = self._extract_price_info(container)
            bid_count = self._extract_bid_count(container)
            image_urls = self.extract_images(container, auction_url)
            end_time, time_remaining = self._extract_item_time_info(container)
            
            item = self.create_auction_item_safely(
                lot_number=lot_number,
                description=description,
                current_price=current_price,
                price_text=price_text,
                bid_count=bid_count,
                source=auction_url,
                auction_id=auction_id,
                end_time=end_time,
                time_remaining=time_remaining,
                image_urls=image_urls,
                auction_title=auction_title,
                company_name="Earl's Auction Company"
            )
            
            return item
            
        except Exception as e:
            logger.error(f"Error parsing item: {e}")
            return None

    def _extract_lot_number(self, container, index):
        """Extract lot number from container"""
        text = container.get_text()
        
        patterns = [
            r'(?:Lot|Item|#)\s*[:\-]?\s*(\d+[a-zA-Z]?)',
            r'(?:^|\s)(\d+[a-zA-Z]?)[:\-]',
            r'#(\d+[a-zA-Z]?)',
            r'(\d{1,4}[a-zA-Z]?)\s*(?:\.|:|\-)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                lot_num = match.group(1)
                if lot_num.isdigit() and 1 <= int(lot_num) <= 9999:
                    return lot_num
                elif len(lot_num) <= 6:
                    return lot_num
        
        return str(index + 1)

    def _extract_description(self, container):
        """Extract item description"""
        desc_selectors = ['.description', '.title', '.name', 'h1', 'h2', 'h3', 'h4', 'strong', 'b']
        
        for selector in desc_selectors:
            element = container.select_one(selector)
            if element:
                text = element.get_text().strip()
                if len(text) > 5 and not re.match(r'^(Lot|#|\d+)', text):
                    return text[:300]
        
        text_parts = [part.strip() for part in container.get_text().split('\n') if part.strip()]
        for part in text_parts:
            if (len(part) > 15 and 
                not re.match(r'^(Lot|#|\d+)', part) and 
                '$' not in part and 
                'bid' not in part.lower()):
                return part[:300]
        
        return "Item description not available"

    def _extract_price_info(self, container):
        """Extract price information from container - FIXED VERSION"""
        logger.debug("Extracting price from container")
        
        # Get text but limit the amount we process to prevent concatenation
        text = container.get_text()
        if len(text) > 500:
            # If text is very long, try to find relevant sections
            sentences = text.split('.')
            relevant_text = ""
            for sentence in sentences:
                if any(keyword in sentence.lower() for keyword in ['bid', 'price', '$']):
                    relevant_text += sentence + ". "
            if relevant_text:
                text = relevant_text
            else:
                text = text[:500]  # Just take first 500 chars as fallback
        
        logger.debug(f"Processing container text: {repr(text[:200])}")
        
        price = self.parse_price(text)
        if price is not None:
            return price, f"${price:.2f}"
        
        return None, ""

    def _extract_bid_count(self, container):
        """Extract bid count"""
        text = container.get_text()
        
        bid_patterns = [
            r'(\d+)\s*Bids?',
            r'Bids?[:\s]*(\d+)',
            r'(\d+)\s*(?:bidders?|bids?)'
        ]
        
        for pattern in bid_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    return int(match.group(1))
                except ValueError:
                    continue
        
        return 0

    def _extract_item_time_info(self, container):
        """Extract time information for individual item"""
        text = container.get_text()
        
        time_elem = container.find(text=re.compile(r'End|Time.*Left|Closing', re.IGNORECASE))
        if time_elem and time_elem.parent:
            time_text = time_elem.parent.get_text()
            return self.parse_time_remaining(time_text)
        
        return None, None

    def scrape_and_store_all(self, include_individual_auctions=True, max_auctions=5):
        """Main method to scrape and store all items in database"""
        logger.info("Starting Earl's Auction scraper...")

        all_items = []
        all_auctions = []

        try:
            auctions = self.scrape_earls_auctions()
            all_auctions.extend(auctions)
            
            if include_individual_auctions:
                processed_auctions = 0
                for auction in auctions:
                    if processed_auctions >= max_auctions:
                        break
                        
                    if auction.company_url:
                        items = self.scrape_auction_details(auction.company_url)
                        all_items.extend(items)
                        processed_auctions += 1
                        time.sleep(2)

            stored_auctions = 0
            stored_items = 0

            for auction in all_auctions:
                try:
                    self.db.save_auction(auction)
                    stored_auctions += 1
                except Exception as e:
                    logger.error(f"Error storing auction: {e}")

            for item in all_items:
                try:
                    self.db.save_item(item)
                    stored_items += 1
                except Exception as e:
                    logger.error(f"Error storing item: {e}")

            logger.info(f"Stored {stored_auctions} auctions and {stored_items} items in database")
            return all_items, all_auctions

        except Exception as e:
            logger.error(f"Error in scrape_and_store_all: {e}")
            return [], []

    def scrape_zip_code_auctions_enhanced(self, max_auctions=10):
        """Enhanced zip code auction scraping"""
        auctions = self.scrape_earls_auctions()
        
        if self.zip_code:
            for auction in auctions:
                auction.zip_code = self.zip_code
        
        return auctions[:max_auctions]

    def scrape_individual_auctions_enhanced(self, max_auctions=5):
        """Enhanced individual auction scraping"""
        auctions = self.scrape_earls_auctions()
        all_items = []
        
        processed = 0
        for auction in auctions:
            if processed >= max_auctions:
                break
                
            if auction.company_url:
                items = self.scrape_auction_details(auction.company_url)
                all_items.extend(items)
                processed += 1
                time.sleep(2)
        
        return all_items

    def scrape_catalog_page_enhanced(self, catalog_url):
        """Enhanced catalog page scraping"""
        return self.scrape_auction_details(catalog_url)


# Helper functions
def get_recent_auctions(db_path='hibid_auctions.db', zip_code=None, hours=24):
    """Get recent auctions from database"""
    db = DatabaseManager(db_path)
    return db.get_active_auctions(zip_code)

def get_auction_items(auction_id, db_path='hibid_auctions.db'):
    """Get all items for a specific auction"""
    db = DatabaseManager(db_path)
    return db.get_items_by_auction(auction_id)

def search_items(query, db_path='hibid_auctions.db', limit=50):
    """Search for items by description"""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT i.*, GROUP_CONCAT(img.image_url) as image_urls
            FROM items i
            LEFT JOIN item_images img ON i.id = img.item_id
            WHERE i.description LIKE ?
            AND datetime(i.scraped_at) > datetime('now', '-24 hours')
            GROUP BY i.id
            ORDER BY i.current_price DESC
            LIMIT ?
        ''', (f'%{query}%', limit))

        columns = [desc[0] for desc in cursor.description]
        items = []
        for row in cursor.fetchall():
            item = dict(zip(columns, row))
            if item['image_urls']:
                item['image_urls'] = item['image_urls'].split(',')
            else:
                item['image_urls'] = []
            items.append(item)
        return items

def scrape_earls_only(zip_code=None, max_auctions=5, db_path='hibid_auctions.db'):
    """Convenience function to scrape only Earl's auctions"""
    scraper = EarlsAuctionScraper(zip_code=zip_code, db_path=db_path)
    return scraper.scrape_and_store_all(
        include_individual_auctions=True,
        max_auctions=max_auctions
    )

def combine_hibid_and_earls(zip_code=None, max_auctions=3, db_path='hibid_auctions.db'):
    """Function to run both Hibid and Earl's scrapers"""
    logger.info("Running combined Hibid + Earl's auction scraper...")
    
    all_items = []
    all_auctions = []
    
    try:
        logger.info("Starting Earl's Auction scraper...")
        earls_scraper = EarlsAuctionScraper(zip_code=zip_code, db_path=db_path)
        earls_items, earls_auctions = earls_scraper.scrape_and_store_all(
            include_individual_auctions=True,
            max_auctions=max_auctions
        )
        all_items.extend(earls_items)
        all_auctions.extend(earls_auctions)
        
        logger.info(f"Combined scraping complete: {len(all_items)} items, {len(all_auctions)} auctions")
        return all_items, all_auctions
        
    except Exception as e:
        logger.error(f"Error in combined scraping: {e}")
        return [], []
