import csv
import os
from sqlalchemy.orm import Session
from .database import SessionLocal, engine, Base
from . import models

# Ensure tables are created
Base.metadata.create_all(bind=engine)

def load_phishing_db():
    db = SessionLocal()
    # Paths relative to project root
    csv_path = "../../MILESTONE_1/data/dataset_phishing.csv"
    
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return

    print(f"Loading data from {csv_path}...")
    
    try:
        with open(csv_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            count = 0
            batch = []
            
            # Check if data already exists to avoid duplicates
            if db.query(models.PhishingURL).first():
                print("Phishing data already exists in database. Skipping import.")
                return

            for row in reader:
                url = row['url']
                status = row['status']
                
                # Check for duplicates in the batch
                phish_url = models.PhishingURL(url=url, status=status)
                batch.append(phish_url)
                
                count += 1
                if count % 1000 == 0:
                    db.bulk_save_objects(batch)
                    db.commit()
                    batch = []
                    print(f"Imported {count} records...")
            
            if batch:
                db.bulk_save_objects(batch)
                db.commit()
            
            print(f"Successfully imported {count} records!")
            
    except Exception as e:
        print(f"An error occurred: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    load_phishing_db()
