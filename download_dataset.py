"""
Download script for ISOT Fake News Dataset
This dataset is real and publicly available for fake news detection research.
Source: https://www.uvic.ca/engineering/ece/isot/index.html
"""

import os
import zipfile
import requests
import pandas as pd
from pathlib import Path

# Dataset URLs (ISOT Fake News Dataset)
DATASET_URLS = {
    "true": "https://www.dropbox.com/s/5y2uzi6p7p23i9z/True.csv?dl=1",
    "fake": "https://www.dropbox.com/s/1p8ym5p4n7n7j9z/Fake.csv?dl=1"
}

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

def download_file(url, dest_path):
    """Download a file from URL"""
    print(f"Downloading {url.split('/')[-1].split('?')[0]}...")
    
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Downloaded to {dest_path}")
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def load_existing_data():
    """Check if data already exists"""
    true_path = DATA_DIR / "True.csv"
    fake_path = DATA_DIR / "Fake.csv"
    
    if true_path.exists() and fake_path.exists():
        print("Loading existing dataset...")
        try:
            true_df = pd.read_csv(true_path)
            fake_df = pd.read_csv(fake_path)
            print(f"Loaded {len(true_df)} true news and {len(fake_df)} fake news articles")
            return true_df, fake_df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None, None
    return None, None

def create_combined_dataset():
    """Create combined dataset from true and fake news"""
    true_df, fake_df = load_existing_data()
    
    if true_df is None:
        print("\nDataset not found locally. Please manually download from:")
        print("True News: https://www.dropbox.com/s/5y2uzi6p7p23i9z/True.csv")
        print("Fake News: https://www.dropbox.com/s/1p8ym5p4n7n7j9z/Fake.csv")
        print("\nOr search for 'ISOT Fake News Dataset' online")
        print("Place the downloaded files in the 'data' folder")
        return None
    
    # Add labels
    true_df['label'] = 1  # 1 = True/Real news
    fake_df['label'] = 0  # 0 = Fake news
    
    # Combine datasets
    df = pd.concat([true_df, fake_df], ignore_index=True)
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save combined dataset
    combined_path = DATA_DIR / "news_dataset.csv"
    df.to_csv(combined_path, index=False)
    
    print(f"\nDataset Summary:")
    print(f"Total articles: {len(df)}")
    print(f"True news: {len(true_df)}")
    print(f"Fake news: {len(fake_df)}")
    print(f"Saved to: {combined_path}")
    
    return df

def create_extended_sample_data():
    """Create an extended sample dataset with more realistic examples"""
    print("\nCreating extended sample dataset for better training...")
    
    # Real news samples - based on actual news topics and writing style
    true_news = [
        # Politics
        {"title": "President Announces New Economic Policy to Boost Growth", "text": "The president announced a new economic policy aimed at reducing unemployment and boosting economic growth. The policy includes tax cuts for small businesses and investments in infrastructure projects. Economic analysts say the measures could increase GDP by 2% over the next year. The proposal will be submitted to Congress for approval next week.", "subject": "politics", "date": "2023-01-15"},
        {"title": "Senate Passes Bipartisan Infrastructure Bill", "text": "The Senate passed a bipartisan infrastructure bill with a vote of 68-32. The legislation includes funding for roads, bridges, broadband internet, and public transit systems. Both Democrats and Republicans supported the measure, which represents one of the largest infrastructure investments in decades.", "subject": "politics", "date": "2023-01-14"},
        {"title": "Congressional Committee Holds Hearing on Climate Policy", "text": "A congressional committee held a hearing on climate policy with testimony from environmental experts, industry representatives, and government officials. The discussion focused on reducing carbon emissions while maintaining economic growth. Lawmakers from both parties expressed support for renewable energy investments.", "subject": "politics", "date": "2023-01-13"},
        {"title": "International Summit Addresses Global Economic Cooperation", "text": "World leaders gathered at an international summit to discuss global economic cooperation and trade policies. The meeting addressed issues including supply chain resilience, currency stability, and sustainable development. Representatives from over 40 countries attended the two-day conference.", "subject": "world news", "date": "2023-01-12"},
        
        # Business
        {"title": "Stock Market Closes Higher Amid Positive Economic Data", "text": "The stock market closed higher today following positive economic data including better than expected jobs report. Technology stocks led the gains, with major companies seeing increases of 3-5%. The Dow Jones Industrial Average rose 200 points, while the S&P 500 gained 1.2%.", "subject": "business", "date": "2023-01-14"},
        {"title": "Tech Company Reports Record Quarterly Earnings", "text": "A leading technology company reported record quarterly earnings, beating analyst expectations. Revenue increased by 20% year over year, reaching $50 billion. The company attributed the growth to strong demand for cloud services and artificial intelligence products.", "subject": "business", "date": "2023-01-11"},
        {"title": "Federal Reserve Maintains Interest Rate Policy", "text": "The Federal Reserve announced it will maintain its current interest rate policy, citing continued progress toward its dual mandate of maximum employment and stable prices. The decision was unanimous among committee members. Economic projections show moderate growth expected for the next year.", "subject": "business", "date": "2023-01-10"},
        {"title": "Major Retail Chain Announces Expansion Plans", "text": "A major retail chain announced plans to expand its operations with 500 new stores over the next three years. The expansion will create approximately 50,000 jobs across the country. The company cited strong consumer demand and successful store performance as reasons for the growth.", "subject": "business", "date": "2023-01-09"},
        {"title": "Automobile Industry Reports Increase in Sales", "text": "The automobile industry reported a 15% increase in vehicle sales compared to last year. Electric vehicles accounted for 25% of total sales, showing continued growth in the EV market. Analysts attribute the increase to improved supply chains and new model releases.", "subject": "business", "date": "2023-01-08"},
        
        # Science
        {"title": "Scientists Discover New Species in Amazon Rainforest", "text": "Scientists have discovered a new species of frog in the Amazon rainforest. The discovery was made during a research expedition in the remote regions of Brazil. The new species has unique physical characteristics that distinguish it from related species. Researchers plan to study its habitat and behavior.", "subject": "science", "date": "2023-01-13"},
        {"title": "NASA Announces New Mars Mission Findings", "text": "NASA announced new findings from its Mars exploration mission. The rover has discovered evidence of ancient water activity on the red planet. Scientists are analyzing rock samples that may contain organic compounds. The findings could shed light on the planet's geological history.", "subject": "science", "date": "2023-01-12"},
        {"title": "Medical Researchers Report Progress on Cancer Treatment", "text": "Medical researchers reported significant progress in cancer treatment at a major conference. Clinical trials show promising results for a new immunotherapy approach. The treatment has shown effectiveness in treating several types of cancer with fewer side effects than traditional chemotherapy.", "subject": "science", "date": "2023-01-11"},
        {"title": "Climate Study Documents Rising Sea Levels", "text": "A comprehensive climate study documents rising sea levels and their potential impact on coastal communities. Researchers analyzed data from satellites and ocean buoys over a 30-year period. The study projects significant changes to coastlines by 2100 if emissions continue at current levels.", "subject": "science", "date": "2023-01-10"},
        
        # Health
        {"title": "Health Officials Report Decline in COVID Cases", "text": "Health officials reported a significant decline in COVID-19 cases this week. The decrease is attributed to vaccination efforts and natural immunity. Hospitalizations have also decreased by 30% compared to last month. Officials urge continued vigilance while praising public health measures.", "subject": "health", "date": "2023-01-10"},
        {"title": "Medical Study Shows Benefits of Exercise on Mental Health", "text": "A new medical study shows that regular exercise significantly improves mental health outcomes. Researchers followed 10,000 participants over five years. Those who exercised at least three times per week reported lower levels of anxiety and depression. The findings support existing recommendations for physical activity.", "subject": "health", "date": "2023-01-09"},
        {"title": "Pharmaceutical Company Reports Positive Vaccine Trial Results", "text": "A pharmaceutical company reported positive results from its latest vaccine trial. The vaccine showed 95% effectiveness in preventing the disease with no serious side effects. The company plans to submit the data to regulatory authorities for approval. Distribution could begin within months.", "subject": "health", "date": "2023-01-08"},
        
        # World News
        {"title": "Global Climate Summit Reaches Agreement on Emissions", "text": "World leaders have reached a new agreement on reducing carbon emissions at the global climate summit. The deal includes commitments from major economies to achieve net-zero emissions by 2050. Financial mechanisms will support developing countries in their transition to clean energy.", "subject": "world news", "date": "2023-01-12"},
        {"title": "International Trade Agreement Signed by Multiple Nations", "text": "An international trade agreement was signed by representatives from 30 countries. The agreement aims to reduce tariffs and improve economic cooperation. Experts predict the deal could increase global trade by 5% over the next decade. Implementation will begin next year.", "subject": "world news", "date": "2023-01-11"},
        {"title": "United Nations Approves Peacekeeping Mission", "text": "The United Nations Security Council approved a new peacekeeping mission to the region. The mission will involve troops from 15 countries and focus on protecting civilians. The council emphasized the importance of stability and humanitarian assistance.", "subject": "world news", "date": "2023-01-10"},
        
        # Technology
        {"title": "Tech Giants Announce Partnership on AI Safety", "text": "Several major technology companies announced a partnership to develop safety standards for artificial intelligence. The collaboration aims to ensure AI systems are developed responsibly. The companies will share research and best practices to address potential risks.", "subject": "technology", "date": "2023-01-11"},
        {"title": "New Satellite Launch Improves Weather Forecasting", "text": "A new weather satellite was successfully launched into orbit today. The satellite will provide more accurate and timely weather data. Meteorologists say it will significantly improve forecasting accuracy, especially for severe weather events. The satellite is expected to operate for at least 10 years.", "subject": "technology", "date": "2023-01-09"},
        {"title": "Cybersecurity Experts Recommend New Protection Measures", "text": "Cybersecurity experts released new recommendations for protecting against online threats. The guidelines include using strong passwords, enabling two-factor authentication, and keeping software updated. Experts warn of increasing sophistication in cyber attacks targeting individuals and businesses.", "subject": "technology", "date": "2023-01-08"},
    ]
    
    # Fake news samples - realistic looking but with common fake news patterns
    fake_news = [
        # Sensationalist/Breaking News
        {"title": "BREAKING: Secret Government Document Reveals Hidden Truth", "text": "A leaked document reveals that government officials have been hiding evidence of alien contact. Sources say the document contains shocking details about extraterrestrial life. The government has refused to comment. Share this before they delete it!", "subject": "news", "date": "2023-01-15"},
        {"title": "SHOCKING: Miracle Cure Found for All Diseases", "text": "Scientists have discovered a miracle cure that can treat all diseases. The cure has been suppressed by pharmaceutical companies because it would eliminate their profits. Big Pharma doesn't want you to know about this simple remedy.", "subject": "health", "date": "2023-01-14"},
        {"title": "URGENT: Famous Celebrity Announces End of World Prediction", "text": "A famous celebrity has predicted the end of the world. According to their calculations, a massive disaster will occur within the next few days. They say the government knows and is preparing bunkers for elites only.", "subject": "news", "date": "2023-01-13"},
        {"title": "EXPOSED: What Food Companies Don't Want You to Know", "text": "Food companies are adding dangerous chemicals to your food. This secret has been hidden from the public for decades. They are poisoning us for profit. Many experts are warning people but the mainstream media won't report it.", "subject": "health", "date": "2023-01-12"},
        
        # Conspiracy Theories
        {"title": "ALERT: Banks About to Collapse, Withdraw Money Now", "text": "Insider sources reveal that major banks are about to collapse. Everyone is advised to withdraw their money immediately to avoid losing everything. The government is hiding the truth about the economy. This could happen any day now!", "subject": "business", "date": "2023-01-14"},
        {"title": "You Won't Believe This Secret About the Moon Landing", "text": "New evidence proves the moon landing was faked. NASA has been lying to us for decades. This footage shows exactly how they created the fake moon landing in a studio. The truth is finally coming out!", "subject": "science", "date": "2023-01-13"},
        {"title": "CONFIRMED: Time Traveler From Future Warns of Disaster", "text": "A person claiming to be a time traveler from the future has appeared on social media warning of an upcoming disaster. They say we have only days to prepare. They have correctly predicted several events in the past.", "subject": "news", "date": "2023-01-12"},
        {"title": "LEAKED: Secret Plan to Control Everyone's Mind", "text": "A leaked government document reveals a secret plan to control everyone's minds through television signals. The program has been running for years and most people don't even know it. They are turning us into zombies!", "subject": "news", "date": "2023-01-11"},
        
        # Medical Misinformation
        {"title": "Doctors Reveal: This Common Food Causes Cancer", "text": "Doctors are hiding the truth about this common food that causes cancer. The government knows and does nothing because food companies pay them billions. Stop eating this immediately or you will get cancer!", "subject": "health", "date": "2023-01-13"},
        {"title": "WARNING: Vaccine Contains Microchips for Tracking", "text": "Shocking truth revealed about vaccines! They contain microscopic chips that track your every move. The government is using vaccines to spy on citizens. This is the end of privacy as we know it!", "subject": "health", "date": "2023-01-12"},
        {"title": "MIRACLE: This Simple Herb Cures All Diseases", "text": "Big Pharma doesn't want you to know about this simple herb that cures cancer, diabetes, and heart disease. Doctors are paid to lie to you. This one weird trick will make all your health problems disappear!", "subject": "health", "date": "2023-01-11"},
        {"title": "EXPOSED: Doctors Recommending Deadly Treatments", "text": "Big Pharma is paying doctors to recommend dangerous treatments instead of cures. They make billions keeping people sick. This doctor reveals the truth about what really works but is being suppressed.", "subject": "health", "date": "2023-01-10"},
        
        # Political Conspiracies
        {"title": "ELECTIONS: Proof That Voting Machines Were Hacked", "text": "Cyber experts have found proof that voting machines were hacked in the last election. The mainstream media refuses to cover this story. The steal was massive and they are trying to cover it up!", "subject": "politics", "date": "2023-01-14"},
        {"title": "SECRET: World Leaders Belong to Secret Cult", "text": "Investigative journalists have uncovered evidence that world leaders belong to a secret cult. They meet in hidden locations and perform rituals. This explains everything wrong with politics today!", "subject": "politics", "date": "2023-01-13"},
        {"title": "REVEALED: Politicians Taking Money from Foreign Governments", "text": "Leaked documents show top politicians taking money from foreign governments. They are selling out their country for personal gain. The corruption goes all the way to the top and no one is safe!", "subject": "politics", "date": "2023-01-12"},
        {"title": "URGENT: New World Order Plan Exposed", "text": "Documents reveal a secret plan for a new world order. Global elites are working together to eliminate national sovereignty. They want to create a one-world government and control everyone!", "subject": "politics", "date": "2023-01-11"},
        
        # Celebrity Gossip/Fake Stories
        {"title": "SHOCKING: Famous Actor Found Dead in Hotel Room", "text": "Tragic news! A famous actor was found dead in a hotel room under mysterious circumstances. Sources say it was not suicide but something much more sinister. The mainstream media is covering up the real cause of death!", "subject": "entertainment", "date": "2023-01-14"},
        {"title": "CELEBRITY SECRET: They're Not Who You Think They Are", "text": "Famous celebrities are not human! This shocking report reveals the truth about their real origins. They've been deceiving the public for years. The evidence is absolutely undeniable!", "subject": "entertainment", "date": "2023-01-13"},
        
        # Scare Tactics
        {"title": "RUNNING OUT: This Essential Resource Will Be Gone in Days", "text": "Scientists warn that this essential resource will be completely gone within days. The government is hiding this from the public. Stock up now before it's too late. This could be the biggest crisis in history!", "subject": "news", "date": "2023-01-12"},
        {"title": "CATASTROPHE: Giant Earthquake Will Destroy Entire Coast", "text": "Seismologists predict a massive earthquake will destroy the entire west coast within weeks. The government knows and isn't telling anyone. Millions will die if you don't prepare now!", "subject": "news", "date": "2023-01-11"},
        {"title": "PANDEMIC 2: New Deadly Disease Spreading Rapidly", "text": "Health experts warn of a new deadly pandemic spreading across the world. The disease is more deadly than COVID and there's no treatment. The government is covering up the true extent of the outbreak!", "subject": "health", "date": "2023-01-10"},
    ]
    
    # Create dataframes
    true_df = pd.DataFrame(true_news)
    fake_df = pd.DataFrame(fake_news)
    
    # Add labels
    true_df['label'] = 1
    fake_df['label'] = 0
    
    # Combine and shuffle
    df = pd.concat([true_df, fake_df], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save
    combined_path = DATA_DIR / "news_dataset.csv"
    df.to_csv(combined_path, index=False)
    
    print(f"Extended dataset created with {len(df)} articles")
    print(f"  Real news: {len(true_df)}")
    print(f"  Fake news: {len(fake_df)}")
    print(f"Saved to: {combined_path}")
    
    return df


if __name__ == "__main__":
    print("=" * 60)
    print("ISOT Fake News Dataset Downloader")
    print("=" * 60)
    
    # Try to load existing data first
    true_df, fake_df = load_existing_data()
    
    if true_df is None:
        # Try downloading
        print("\nAttempting to download dataset...")
        
        # Try download true news
        true_path = DATA_DIR / "True.csv"
        fake_path = DATA_DIR / "Fake.csv"
        
        success = True
        if not true_path.exists():
            success = download_file(DATASET_URLS["true"], true_path) and success
        if not fake_path.exists():
            success = download_file(DATASET_URLS["fake"], fake_path) and success
        
        if success:
            create_combined_dataset()
        else:
            print("\nCould not download dataset. Creating extended sample data...")
            create_extended_sample_data()
    else:
        create_combined_dataset()
    
    print("\nDone!")

