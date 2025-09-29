from bs4 import BeautifulSoup
import requests
import csv
import time

def script_call_participants():
    print(">>> Script start <<<", flush=True)
    input_csv_path = r'C:\Users\nmrva\OneDrive\Desktop\AI_CHIP_STRATEGY\Data\CSV\amd_earnings_calls.csv'
    output_csv_path = r'C:\Users\nmrva\OneDrive\Desktop\AI_CHIP_STRATEGY\Data\CSV\amd_call_participants.csv'
    info_csv_path = r'C:\Users\nmrva\OneDrive\Desktop\AI_CHIP_STRATEGY\Data\CSV\amd_call_employees.csv'

    with open(info_csv_path, newline = '', encoding = 'utf-8') as out_f:
        info_reader = csv.DictReader(out_f)
        existing_names = set()
        for row in info_reader:
            existing_names.add(row.get('Name'))

    with open(output_csv_path, mode = 'w', newline = '', encoding = 'utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["Name", "Title", "Transcript", "Source URL"])

        with open(input_csv_path, newline = '', encoding = 'utf-8') as infile:
            reader = csv.DictReader(infile)
            
            seen_names = set()
            
            for row in reader:
                page_title = row.get('Title')
                url = row['URL']
                if not url or not page_title:
                    continue

                try:
                    response = requests.get(url, timeout = 2)
                    html = response.text 

                except Exception as e:
                    print(f" Request failed for {url}: {e}")
                
                if response.status_code == 200:
                    print(f"Got Status {response.status_code} for {url}")

                    soup = BeautifulSoup(html, 'html.parser')

                    call_participants_header = soup.find(lambda tag: tag.name in ['h2', 'h3', 'h4'] and 'call participants' in tag.get_text(strip=True).lower())

                    if call_participants_header:

                        p_tags = call_participants_header.find_next_siblings('p')
                        for p in p_tags:
                            name_tag = p.find('strong')
                            title_tag = p.find('em')

                            if name_tag and title_tag:
                                name = name_tag.get_text(strip = True) 
                                title = title_tag.get_text(strip = True)

                            if name not in existing_names:
                                writer.writerow([name, title, page_title, url])

if __name__ == "__main__":
    script_call_participants()