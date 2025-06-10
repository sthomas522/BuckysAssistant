#!/usr/bin/env python3
"""
Database Upgrade Script for Enhanced Auction Features
Adds necessary columns and improves existing data
"""

import sqlite3
import logging
from datetime import datetime
from typing import List, Dict
import re

logger = logging.getLogger(__name__)

class DatabaseUpgrader:
    """Handles database schema upgrades and data improvements"""
    
    def __init__(self, db_path: str = 'auction_data.db'):
        self.db_path = db_path
        
    def upgrade_database(self):
        """Perform all necessary database upgrades"""
        logger.info("Starting database upgrade...")
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Check current schema version
            try:
                cursor.execute("SELECT version FROM schema_version ORDER BY id DESC LIMIT 1")
                current_version = cursor.fetchone()
                current_version = current_version[0] if current_version else 0
            except sqlite3.OperationalError:
                # Table doesn't exist, create it
                cursor.execute("""
                    CREATE TABLE schema_version (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        version INTEGER,
                        upgrade_date TEXT,
                        description TEXT
                    )
                """)
                current_version = 0
            
            logger.info(f"Current database version: {current_version}")
            
            # Apply upgrades
            if current_version < 1:
                self._upgrade_to_v1(cursor)
            if current_version < 2:
                self._upgrade_to_v2(cursor)
            if current_version < 3:
                self._upgrade_to_v3(cursor)
                
            conn.commit()
            logger.info("Database upgrade completed")
    
    def _upgrade_to_v1(self, cursor):
        """Upgrade to version 1 - Add missing columns"""
        logger.info("Upgrading to database version 1...")
        
        # Add columns to items table if they don't exist
        new_columns = [
            ("item_url", "TEXT"),  # Direct URL to the item
            ("estimated_value", "REAL"),  # Estimated value if available
            ("condition_notes", "TEXT"),  # Item condition
            ("category", "TEXT"),  # Item category
        ]
        
        for column_name, column_type in new_columns:
            try:
                cursor.execute(f"ALTER TABLE items ADD COLUMN {column_name} {column_type}")
                logger.info(f"Added column {column_name} to items table")
            except sqlite3.OperationalError as e:
                if "duplicate column name" in str(e).lower():
                    logger.debug(f"Column {column_name} already exists")
                else:
                    logger.error(f"Error adding column {column_name}: {e}")
        
        # Add columns to auctions table if they don't exist
        auction_columns = [
            ("preview_date", "TEXT"),  # Preview date
            ("pickup_date", "TEXT"),   # Pickup date
            ("terms_url", "TEXT"),     # Terms and conditions URL
            ("auctioneer_info", "TEXT"), # Auctioneer information
        ]
        
        for column_name, column_type in auction_columns:
            try:
                cursor.execute(f"ALTER TABLE auctions ADD COLUMN {column_name} {column_type}")
                logger.info(f"Added column {column_name} to auctions table")
            except sqlite3.OperationalError as e:
                if "duplicate column name" in str(e).lower():
                    logger.debug(f"Column {column_name} already exists")
                else:
                    logger.error(f"Error adding column {column_name}: {e}")
        
        # Record version upgrade
        cursor.execute("""
            INSERT INTO schema_version (version, upgrade_date, description)
            VALUES (1, ?, 'Added missing columns for enhanced features')
        """, (datetime.now().isoformat(),))
    
    def _upgrade_to_v2(self, cursor):
        """Upgrade to version 2 - Improve time parsing"""
        logger.info("Upgrading to database version 2...")
        
        # Get all items with 'Unknown' time remaining
        cursor.execute("""
            SELECT id, time_remaining, end_time, source, auction_id, lot_number
            FROM items 
            WHERE time_remaining = 'Unknown' OR time_remaining IS NULL
        """)
        
        items_to_update = cursor.fetchall()
        logger.info(f"Found {len(items_to_update)} items with unknown time remaining")
        
        # Try to parse time remaining from end_time
        from enhanced_time_parser import EnhancedTimeParser
        
        updated_count = 0
        for item_id, time_remaining, end_time, source, auction_id, lot_number in items_to_update:
            new_end_time, new_time_remaining = EnhancedTimeParser.parse_time_remaining(
                time_remaining, end_time
            )
            
            if new_time_remaining and new_time_remaining != "Unknown":
                cursor.execute("""
                    UPDATE items 
                    SET time_remaining = ?, end_time = ?
                    WHERE id = ?
                """, (new_time_remaining, new_end_time or end_time, item_id))
                updated_count += 1
        
        logger.info(f"Updated time remaining for {updated_count} items")
        
        # Record version upgrade
        cursor.execute("""
            INSERT INTO schema_version (version, upgrade_date, description)
            VALUES (2, ?, 'Improved time parsing for existing items')
        """, (datetime.now().isoformat(),))
    
    def _upgrade_to_v3(self, cursor):
        """Upgrade to version 3 - Add item URLs and improve data quality"""
        logger.info("Upgrading to database version 3...")
        
        # Update item_url for items that don't have it
        cursor.execute("""
            SELECT id, source, auction_id, lot_number
            FROM items 
            WHERE item_url IS NULL OR item_url = ''
        """)
        
        items_to_update = cursor.fetchall()
        logger.info(f"Found {len(items_to_update)} items without direct URLs")
        
        from enhanced_time_parser import EnhancedDataExtractor
        
        updated_count = 0
        for item_id, source, auction_id, lot_number in items_to_update:
            item_url = EnhancedDataExtractor.create_item_link(source, auction_id, lot_number)
            if item_url and item_url != source:
                cursor.execute("UPDATE items SET item_url = ? WHERE id = ?", (item_url, item_id))
                updated_count += 1
        
        logger.info(f"Added direct URLs for {updated_count} items")
        
        # Clean up descriptions - remove "No description" and improve quality
        cursor.execute("""
            UPDATE items 
            SET description = 'Item details not available'
            WHERE description IN ('No description', 'No description available', '')
        """)
        
        # Record version upgrade
        cursor.execute("""
            INSERT INTO schema_version (version, upgrade_date, description)
            VALUES (3, ?, 'Added item URLs and improved data quality')
        """, (datetime.now().isoformat(),))
    
    def validate_and_fix_data(self):
        """Validate and fix common data issues"""
        logger.info("Validating and fixing data issues...")
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Fix price issues
            self._fix_price_issues(cursor)
            
            # Fix time remaining issues
            self._fix_time_issues(cursor)
            
            # Fix description issues
            self._fix_description_issues(cursor)
            
            # Remove duplicate items
            self._remove_duplicates(cursor)
            
            conn.commit()
    
    def _fix_price_issues(self, cursor):
        """Fix common price-related issues"""
        # Fix negative prices
        cursor.execute("UPDATE items SET current_price = NULL WHERE current_price < 0")
        
        # Fix unreasonably high prices (over $100,000)
        cursor.execute("UPDATE items SET current_price = NULL WHERE current_price > 100000")
        
        # Update price_text for items with fixed prices
        cursor.execute("""
            UPDATE items 
            SET price_text = '$' || printf('%.2f', current_price)
            WHERE current_price IS NOT NULL AND current_price > 0
        """)
        
        cursor.execute("""
            UPDATE items 
            SET price_text = 'No bids'
            WHERE current_price IS NULL OR current_price = 0
        """)
        
        logger.info("Fixed price issues")
    
    def _fix_time_issues(self, cursor):
        """Fix time-related issues"""
        from enhanced_time_parser import EnhancedTimeParser
        
        # Get items with problematic time remaining
        cursor.execute("""
            SELECT id, time_remaining, end_time 
            FROM items 
            WHERE time_remaining IN ('Unknown', '', NULL) 
            AND datetime(scraped_at) > datetime('now', '-7 days')
        """)
        
        items = cursor.fetchall()
        updated = 0
        
        for item_id, time_remaining, end_time in items:
            new_end_time, new_time_remaining = EnhancedTimeParser.parse_time_remaining(
                time_remaining, end_time
            )
            
            if new_time_remaining and new_time_remaining != "Unknown":
                cursor.execute("""
                    UPDATE items 
                    SET time_remaining = ?, end_time = ?
                    WHERE id = ?
                """, (new_time_remaining, new_end_time or end_time, item_id))
                updated += 1
        
        logger.info(f"Fixed time issues for {updated} items")
    
    def _fix_description_issues(self, cursor):
        """Fix description-related issues"""
        # Clean up empty descriptions
        cursor.execute("""
            UPDATE items 
            SET description = 'Item details not available'
            WHERE description IS NULL 
            OR description = '' 
            OR description = 'No description'
            OR description = 'No description available'
        """)
        
        # Truncate overly long descriptions
        cursor.execute("""
            UPDATE items 
            SET description = substr(description, 1, 300) || '...'
            WHERE length(description) > 300
        """)
        
        logger.info("Fixed description issues")
    
    def _remove_duplicates(self, cursor):
        """Remove duplicate items based on lot_number, auction_id, and source"""
        cursor.execute("""
            DELETE FROM items 
            WHERE id NOT IN (
                SELECT MIN(id) 
                FROM items 
                GROUP BY lot_number, auction_id, source
            )
        """)
        
        removed = cursor.rowcount
        logger.info(f"Removed {removed} duplicate items")
    
    def get_database_stats(self) -> Dict:
        """Get database statistics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            stats = {}
            
            # Count items
            cursor.execute("SELECT COUNT(*) FROM items")
            stats['total_items'] = cursor.fetchone()[0]
            
            # Count recent items
            cursor.execute("""
                SELECT COUNT(*) FROM items 
                WHERE datetime(scraped_at) > datetime('now', '-24 hours')
            """)
            stats['recent_items'] = cursor.fetchone()[0]
            
            # Count auctions
            cursor.execute("SELECT COUNT(*) FROM auctions")
            stats['total_auctions'] = cursor.fetchone()[0]
            
            # Count items with images
            cursor.execute("""
                SELECT COUNT(DISTINCT item_id) FROM item_images
            """)
            stats['items_with_images'] = cursor.fetchone()[0]
            
            # Count items with prices
            cursor.execute("""
                SELECT COUNT(*) FROM items 
                WHERE current_price > 0
            """)
            stats['items_with_prices'] = cursor.fetchone()[0]
            
            # Count items by company
            cursor.execute("""
                SELECT company_name, COUNT(*) 
                FROM items 
                WHERE datetime(scraped_at) > datetime('now', '-24 hours')
                GROUP BY company_name
            """)
            stats['items_by_company'] = dict(cursor.fetchall())
            
            # Get schema version
            try:
                cursor.execute("SELECT MAX(version) FROM schema_version")
                stats['schema_version'] = cursor.fetchone()[0] or 0
            except sqlite3.OperationalError:
                stats['schema_version'] = 0
            
            return stats


def main():
    """Main function to run database upgrade"""
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Upgrade auction database')
    parser.add_argument('--db-path', default='auction_data.db', help='Database path')
    parser.add_argument('--validate-only', action='store_true', help='Only validate and fix data, no schema changes')
    parser.add_argument('--stats', action='store_true', help='Show database statistics')
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    upgrader = DatabaseUpgrader(args.db_path)
    
    if args.stats:
        stats = upgrader.get_database_stats()
        print("\n📊 Database Statistics:")
        print("-" * 30)
        print(f"Schema Version: {stats['schema_version']}")
        print(f"Total Items: {stats['total_items']}")
        print(f"Recent Items (24h): {stats['recent_items']}")
        print(f"Total Auctions: {stats['total_auctions']}")
        print(f"Items with Images: {stats['items_with_images']}")
        print(f"Items with Prices: {stats['items_with_prices']}")
        print("\nItems by Company:")
        for company, count in stats['items_by_company'].items():
            print(f"  • {company}: {count}")
        return
    
    if args.validate_only:
        upgrader.validate_and_fix_data()
    else:
        upgrader.upgrade_database()
        upgrader.validate_and_fix_data()
    
    # Show final stats
    stats = upgrader.get_database_stats()
    print(f"\n✅ Database upgrade completed!")
    print(f"Schema version: {stats['schema_version']}")
    print(f"Total items: {stats['total_items']}")
    print(f"Recent items: {stats['recent_items']}")


if __name__ == "__main__":
    main()