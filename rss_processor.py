import feedparser
import json
import datetime
import time
import os
import sys
from bs4 import BeautifulSoup
from openai import OpenAI
import re
from supabase import create_client, Client

def get_openai_api_key():
    """從環境變量中獲取OpenAI API密鑰"""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables. Please set the OPENAI_API_KEY environment variable.")
    return api_key

try:
    client = OpenAI(api_key=get_openai_api_key())
except ValueError as e:
    print(f"Error: {e}")
    print("Please make sure to set the OPENAI_API_KEY environment variable before running this script.")
    sys.exit(1)

# Initialize Supabase client
url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)

def preprocess_content(text):
    """預處理文本內容，移除不必要的部分"""
    text = re.sub(r'^.*?(?=ABSTRACT|OBJECTIVES)', '', text, flags=re.DOTALL)
    text = re.sub(r'\s*PMID:.*$', '', text, flags=re.DOTALL)
    return text.strip()

def translate_title(text, target_language="zh-TW"):
    """使用OpenAI API翻譯文章標題"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"""You are a translator specializing in academic article titles. Translate the following title to {target_language}. Ensure the translation is concise and accurate, maintaining any technical terms. Use Traditional Chinese (Taiwan) and avoid using Simplified Chinese."""},
                {"role": "user", "content": text}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error in translate_title: {e}")
        return text

def generate_tldr(text, target_language="zh-TW"):
    """使用OpenAI API生成文章的TL;DR摘要"""
    try:
        preprocessed_text = preprocess_content(text)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"""You are an expert in summarizing academic research. Create an extremely concise TL;DR (Too Long; Didn't Read) summary in {target_language} of the following academic abstract. Follow these guidelines:

1. Summarize the entire abstract in 3-4 short, clear sentences.
2. Focus only on the most crucial information: the main objective, key method, and primary finding or conclusion.
3. Use simple, clear language while maintaining academic accuracy.
4. Start the summary with the emoji 💡 followed by "TL;DR: ".
5. Do not use separate headings or multiple paragraphs.
6. Ensure the summary is written in Traditional Chinese (Taiwan) and avoid using Simplified Chinese.

Ensure the summary captures the essence of the research while being extremely concise."""},
                {"role": "user", "content": preprocessed_text}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error in generate_tldr: {e}")
        return text

def generate_keywords(title, full_content):
    """使用OpenAI API生成文章的關鍵字"""
    try:
        preprocessed_content = preprocess_content(full_content)
        input_text = f"Title: {title}\n\nContent: {preprocessed_content}"
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert in academic content analysis. Generate 5 relevant keywords in English for the given academic article. Focus on the main topics, methods, and findings. Pay special attention to the title as it often contains key information. Separate keywords with commas."},
                {"role": "user", "content": input_text}
            ]
        )
        return response.choices[0].message.content.strip().split(', ')
    except Exception as e:
        print(f"Error in generate_keywords: {e}")
        return []

def fetch_rss_basic(url):
    """獲取 RSS feed 的基本內容"""
    feed = feedparser.parse(url)
    entries = []
    for entry in feed.entries:
        content = entry.get('content', [{}])[0].get('value', '')
        if not content:
            content = entry.get('summary', '')
        
        soup = BeautifulSoup(content, 'html.parser')
        text_content = soup.get_text(separator='\n', strip=True)
        
        pmid = entry['guid'].split(':')[-1] if 'guid' in entry else None
        published = entry.get('published', datetime.datetime.now().isoformat())
        
        # 擷取 DOI
        doi = None
        if 'dc_identifier' in entry:
            for identifier in entry.dc_identifier:
                if identifier.startswith('doi:'):
                    doi = identifier.split('doi:')[-1]
                    break
        
        print(f"Fetched entry - PMID: {pmid}, DOI: {doi}")  # 日誌輸出
        
        entries.append({
            'title': entry.title,
            'link': entry.link,
            'published': published,
            'full_content': text_content,
            'pmid': pmid,
            'doi': doi
        })
    
    return {
        'feed_title': feed.feed.title,
        'feed_link': feed.feed.link,
        'feed_updated': feed.feed.get('updated', datetime.datetime.now().isoformat()),
        'entries': entries
    }

def load_existing_data_for_source(source):
    """從Supabase加載特定源的現有數據"""
    response = supabase.table("rss_entries").select("*").eq("source", source).execute()
    return response.data

def save_rss_data(source, entries):
    """將RSS源的數據保存到Supabase"""
    for entry in entries:
        try:
            existing = supabase.table("rss_entries").select("*").eq("source", source).eq("pmid", entry['pmid']).execute()
            
            if existing.data:
                existing_doi = existing.data[0].get('doi')
                new_doi = entry.get('doi')
                print(f"Existing entry - PMID: {entry['pmid']}, Existing DOI: {existing_doi}, New DOI: {new_doi}")  # 日誌
                
                # 如果新的DOI存在，且與現有DOI不同（包括現有DOI為None的情況），則更新
                if new_doi and new_doi != existing_doi:
                    supabase.table("rss_entries").update({
                        "doi": new_doi
                    }).eq("source", source).eq("pmid", entry['pmid']).execute()
                    print(f"Updated DOI for existing entry {entry['pmid']} for source {source}")
                else:
                    print(f"No DOI update needed for entry {entry['pmid']}")  # 日誌
            else:
                # 對於新條目，插入所有字段，包括 DOI
                supabase.table("rss_entries").insert({
                    "source": source,
                    "title": entry['title'],
                    "title_translated": entry.get('title_translated', ''),
                    "link": entry['link'],
                    "published": entry['published'],
                    "tldr": entry.get('tldr', ''),
                    "pmid": entry['pmid'],
                    "keywords": entry.get('keywords', []),
                    "doi": entry.get('doi')
                }).execute()
                print(f"Inserted new entry {entry['pmid']} for source {source}")
        except Exception as e:
            print(f"Error processing entry {entry['pmid']} for source {source}: {e}")
            print(f"Entry data: {entry}")

def process_rss_sources(sources):
    """處理所有RSS來源並立即保存數據"""
    for name, url in sources.items():
        try:
            print(f"Processing source: {name}")
            new_feed_data = fetch_rss_basic(url)
            existing_entries = load_existing_data_for_source(name)
            existing_pmids = {entry['pmid']: entry for entry in existing_entries if 'pmid' in entry}
            
            new_entries = []
            updated_entries = []
            for entry in new_feed_data['entries']:
                if entry['pmid'] not in existing_pmids:
                    # 處理新文章
                    entry['title_translated'] = translate_title(entry['title'])
                    entry['tldr'] = generate_tldr(entry['full_content'])
                    entry['keywords'] = generate_keywords(entry['title'], entry['full_content'])
                    new_entries.append(entry)
                    print(f"New entry found - PMID: {entry['pmid']}, DOI: {entry.get('doi')}")  # 日誌
                else:
                    # 對於重複文章，檢查是否需要更新 DOI
                    existing_entry = existing_pmids[entry['pmid']]
                    existing_doi = existing_entry.get('doi')
                    new_doi = entry.get('doi')
                    
                    print(f"Existing entry - PMID: {entry['pmid']}, Existing DOI: {existing_doi}, New DOI: {new_doi}")  # 日誌
                    
                    # 如果新的DOI不為空，且與現有DOI不同（包括現有DOI為None的情況）
                    if new_doi and new_doi != existing_doi:
                        existing_entry['doi'] = new_doi
                        updated_entries.append(existing_entry)
                        print(f"Updating DOI for entry {entry['pmid']} from {existing_doi} to {new_doi}")
                    else:
                        print(f"No DOI update needed for entry {entry['pmid']}")  # 日誌
            
            # 合併新文章和需要更新 DOI 的文章
            entries_to_save = new_entries + updated_entries
            
            if entries_to_save:
                save_rss_data(name, entries_to_save)
                print(f"Processed {len(new_entries)} new entries and updated DOIs for {len(updated_entries)} existing entries for {name}")
            else:
                print(f"No new entries or DOI updates for {name}")
        except Exception as e:
            print(f"Error processing source {name}: {e}")
            continue

def load_rss_sources(file_path='rss_sources.json'):
    """從JSON文件加載RSS來源"""
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: RSS sources file '{file_path}' not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in RSS sources file '{file_path}'.")
        sys.exit(1)

if __name__ == "__main__":
    # 主程序
    rss_sources = load_rss_sources()
    
    try:
        process_rss_sources(rss_sources)
        print("RSS data processing completed successfully")
    except Exception as e:
        print(f"An error occurred during RSS processing: {e}")
        sys.exit(1)
