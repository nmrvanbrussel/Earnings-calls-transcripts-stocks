from bs4 import BeautifulSoup, NavigableString, Tag
import requests
import csv
from typing import Optional, Tuple, List, Set

TRANSCRIPTS_CSV = r'C:\Users\nmrva\OneDrive\Desktop\AI_CHIP_STRATEGY\Data\CSV\amd_earnings_calls.csv'
EMPLOYEES_CSV   = r'C:\Users\nmrva\OneDrive\Desktop\AI_CHIP_STRATEGY\Data\CSV\amd_call_employees.csv'
PARTICIPANTS_CSV= r'C:\Users\nmrva\OneDrive\Desktop\AI_CHIP_STRATEGY\Data\CSV\amd_call_participants_test.csv'
OUTPUT_CSV      = r'C:\Users\nmrva\OneDrive\Desktop\AI_CHIP_STRATEGY\Data\CSV\amd_qna_turns.csv'
COMPANY_NAME    = 'Advanced Micro Devices'  # change to NVIDIA when doing NVIDIA


QNA_HEADER_PATTERNS = [
    'Questions & Answers'
]

OPERATOR_TOKENS = {'operator', 'moderator'}

print(f'Start')

def load_names(path: str) -> Set[str]:
    try:
        with open(path, newline = '', encoding = 'utf-8') as f:
            reader = csv.DictReader(f)
            return { (r.get('Name') or '').strip() for r in reader if (r.get('Name'))}
    except FileNotFoundError:
        return set()
    
def find_qna_header(soup: BeautifulSoup) -> Optional[Tag]:
    for tag in soup.find_all(['h2']):
        txt = tag.get_text(strip = True).lower()
        if 'questions & answers:' in txt:
            return tag
    return None
    
def iterate_qna_blocks(header: Tag) -> List[Tag]:
    blocks = []
    for sib in header.next_siblings:
        if isinstance(sib, NavigableString):
            continue
        if isinstance(sib, Tag) and sib.name in ['h2', 'h3', 'h4']:
            break 
        if isinstance(sib, Tag):
            blocks.append(sib)
    return blocks

#A new question is started with <p><strong>Name</strong> <em>Title</em></p>
#The problem of the TUrn based QNA is here as we need to scrape for the OPERATORS ASWELL.
def extract_speaker_from_p (p: Tag) -> Optional[Tuple[str, str, str]]:
    strong = p.find('strong')
    em = p.find('em')
    if not strong or not em:
        return None
    if len(p.find_all('strong')) != 1 or len(p.find_all('em')) != 1:
        return None 
    
    name = strong.get_text(strip = True)
    title = em.get_text(strip = True)
    sp_soup = BeautifulSoup(str(p), 'html.parser')
    for t in sp_soup.find_all(['strong', 'em']):
        t.extract()
    spoken = sp_soup.get_text(separator = "--", strip = True)

    return (name, title, spoken) 

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

def classify_role(name: str, title: str, employees: Set[str], participants: Set[str]) -> str:
    nlow = (name or '').lower() 
    tlow = (title or '').lower()

    if nlow in (nm.lower() for nm in employees): #Limitation that if a past company employee starts asking questions as a non-employee
        return 'company'
    if any(tok in nlow or tok in tlow for tok in OPERATOR_TOKENS):
        return 'operator'
    if nlow in (nm.lower() for nm in participants) or 'analyst' in tlow: # Could polish up the second or, with more titles
        return 'analyst'
    if any(k in tlow for k in accepted_titles) or (COMPANY_NAME.lower().split()[0] in tlow): #Double check
        return 'company'
    return 'other'

print(f'scrape_qna_turns')

def scrape_qna_turns():
    print(f'SCRIPT STARTED')
    employees = load_names(EMPLOYEES_CSV)
    participants = load_names(PARTICIPANTS_CSV)

    with open(OUTPUT_CSV, 'w', newline = '', encoding = 'utf-8') as out_f:
        writer = csv.writer(out_f)

        writer.writerow([
            'Company', 'Call Title', 'Publication Date', 'Quarter End Date',
            'Pair Index', 'Turn Index', 'Speaker Name', 'Speaker Title', 'Speaker Role', 'Text', 'Source URL'
        ])

        with open(TRANSCRIPTS_CSV, newline = '', encoding = 'utf-8') as in_f:
            rdr = csv.DictReader(in_f)
            for row in rdr:
                title = row.get('Title') 
                pub_date = row.get('Publication Date') 
                quarter_end = row.get('Quarter End Date') 
                url = row.get('URL')
                if not url:
                    continue 

                try:
                        resp = requests.get(url, timeout = 5)
                except Exception as e:
                        continue
                if resp.status_code != 200:
                        print(f'HTTP {resp.status_code} {url}')
                        continue
                soup = BeautifulSoup(resp.text, 'html.parser')
                qna_header = find_qna_header(soup)
                if not qna_header:
                    print(f'No Q&A header found: {url}')
                    continue
                    
                blocks = iterate_qna_blocks(qna_header)
                current = None 
                turns = []

                for block in blocks:
                    if block.name != 'p':
                        continue
                    sp = extract_speaker_from_p(block)
                    if sp:
                        if current:
                            turns.append(current)
                        name, stitle, spoken0 = sp
                        role = classify_role(name, stitle, employees, participants)
                        text_parts = [spoken0] if spoken0 else []
                        current = [name, stitle, role, text_parts]
                    else:
                        # Continuation of prior speaker
                        if current:
                            current[3].append(block.get_text(separator=' ', strip=True))

                                
                if current:
                    turns.append(current)
                
                # TURN-BASED LOGIC: Each turn gets a pair_index and turn_index
                pair_index = 1
                turn_index = 1
                
                for turn in turns:
                    print(f"DEBUG: Turn role: '{turn[2]}")
                    if turn[2] == 'operator':
                        print(f"Operator found!")
                        pair_index += 1
                        turn_index = 1  # Reset turn index for new pair
                    
                    text = ' '.join([p for p in turn[3] if p]).strip()
                    if text:
                        writer.writerow([
                            COMPANY_NAME, title, pub_date, quarter_end,
                            pair_index, turn_index,
                            turn[0], turn[1], turn[2], text, url
                        ])
                        turn_index += 1

if __name__ == '__main__':
    print('QNA Scraping Start')
    scrape_qna_turns()
