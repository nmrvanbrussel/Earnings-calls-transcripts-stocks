from bs4 import BeautifulSoup
import requests
import csv
import time

def script_scraping_employees():
    print(">>> Script start <<<", flush=True)
    input_csv_path = r'C:\Users\nmrva\OneDrive\Desktop\AI_CHIP_STRATEGY\Data\CSV\nvidia_earnings_calls.csv'
    output_csv_path = r'C:\Users\nmrva\OneDrive\Desktop\AI_CHIP_STRATEGY\Data\CSV\nvidia_call_employees.csv'

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

                    call_participants_header = soup.find(lambda tag: tag.name in ['h2'] and 'call participants' in tag.get_text(strip=True).lower()) #NEED TO FIX AS IT DOES NOT RETURN ALL EMPLOYEES EVERY TIME
                            

                    if call_participants_header:
                    
                        p_tags = call_participants_header.find_next_siblings('p')        
                        for p in p_tags:
                            name_tag = p.find('strong')
                            title_tag = p.find('em')

                            if name_tag and title_tag:
                                name = name_tag.get_text(strip = True)
                                title = title_tag.get_text(strip = True)

                                if title in accepted_titles:
                                    print(f" Writing: {name} — {title}")
                                    writer.writerow([name, title, page_title, url])
                                        
        print("DONE")

if __name__ == "__main__":
    script_scraping_employees()