from bs4 import BeautifulSoup
import requests
import csv
import time
import re

def script_scraping_employees():
    print(">>> Script start <<<", flush=True)
    input_csv_path = r'C:\Users\nmrva\OneDrive\Desktop\AI_CHIP_STRATEGY\Data\CSV\amd_earnings_calls.csv'
    output_csv_path = r'C:\Users\nmrva\OneDrive\Desktop\AI_CHIP_STRATEGY\Data\CSV\amd_call_participants_test.csv'

 #Extracting Employees
 
    accepted_titles = {
        "President and Chief Executive Officer",
        "Executive Vice President and Chief Financial Officer",
        "Senior Vice President",
        "Vice President, Investor Relations",
        "Chief Technology Officer",
        "Chief Operating Officer",
        "Chief Financial Officer",
        "Founder and CEO",
        "Founder and Chief Executive Officer",
        "General Counsel",
        "Chief Scientist",
        "Head of Investor Relations",
        "Senior Director",
        "Investor Relations"
                        # Add more corporate titles as needed
        }   

    print(f"Opening output file for writing: {output_csv_path}")
    with open(output_csv_path, mode = 'w', newline = '', encoding = 'utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["Name", "Title","Transcript", "Source URL"])

        print(f"Reading input URLs from: {input_csv_path}")
        with open(input_csv_path, newline = '', encoding = 'utf-8') as infile:
                    reader = csv.DictReader(infile)
                    for row in reader:
                        page_title = row.get('Title')
                        print("Loop iteration start")
                        url = row['URL']
                        if not url or not page_title:
                            continue

                        print(f"Fetching: {url}")
                        try:
                            response = requests.get(url, timeout = 2)
                            html = response.text 

                        except Exception as e:
                            print(f" Request failed for {url}: {e}")
                            continue 

                        if response.status_code == 200:
                            print(f"Got Status {response.status_code} for {url}")

                            soup = BeautifulSoup(html, 'html.parser')

                            target_names = []
                            seen_names = set()
                            seen_titles = set()

                            for p in soup.find_all('p'):
                                strong = p.find('strong')
                                em = p.find('em')

                                # Must have exactly one of each
                                if strong and em:
                                        if len(p.find_all('strong')) == 1 and len(p.find_all('em')) == 1:
                                            name = strong.get_text(strip = True)
                                            title = em.get_text(strip = True)
                                    
                                            # Deduplicate based on both name and title
                                            if not any(role.lower() in title.lower() for role in accepted_titles) and name not in seen_names and title not in seen_titles:
                                                writer.writerow([name, title, page_title, url])
                                                seen_names.add(name)
                                                seen_titles.add(title)

                                            for name, title in target_names:
                                                print(f"{name} — {title}")

if __name__ == "__main__":
    script_scraping_employees()