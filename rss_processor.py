import feedparser
import json
import datetime
import time
import os
import sys
import re
from bs4 import BeautifulSoup
from openai import OpenAI
from supabase import create_client, Client

class LiteratureProcessor:
    def __init__(self):
        """初始化文獻處理器"""
        # 初始化 OpenAI 客戶端
        self.api_key = self.get_openai_api_key()
        self.client = OpenAI(api_key=self.api_key)
        self.model = "gpt-4o-mini"
        
        # 向量嵌入設定
        self.enable_embeddings = True
        self.embedding_model = "text-embedding-3-small"
        self.embedding_strategy = "hybrid"
        
        # 初始化 Supabase 客戶端
        url: str = os.environ.get("SUPABASE_URL")
        key: str = os.environ.get("SUPABASE_KEY")
        self.supabase: Client = create_client(url, key)
        
        print(f"✅ LiteratureProcessor 初始化完成")
        print(f"   - OpenAI 模型: {self.model}")
        print(f"   - 嵌入模型: {self.embedding_model}")
        print(f"   - 嵌入策略: {self.embedding_strategy}")

    def get_openai_api_key(self):
        """從環境變量中獲取OpenAI API密鑰"""
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables. Please set the OPENAI_API_KEY environment variable.")
        return api_key

    def preprocess_content(self, text):
        """預處理文本內容，移除不必要的部分"""
        text = re.sub(r'^.*?(?=ABSTRACT|OBJECTIVES)', '', text, flags=re.DOTALL)
        text = re.sub(r'\s*PMID:.*$', '', text, flags=re.DOTALL)
        return text.strip()

    def translate_title(self, text, target_language="zh-TW"):
        """使用OpenAI API翻譯文章標題"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": f"""You are a translator specializing in academic article titles. Translate the following title to {target_language}. Ensure the translation is concise and accurate, maintaining any technical terms. Use Traditional Chinese (Taiwan) and avoid using Simplified Chinese."""},
                    {"role": "user", "content": text}
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error in translate_title: {e}")
            return text

    def generate_english_tldr(self, text):
        """生成英文TL;DR摘要"""
        try:
            preprocessed_text = self.preprocess_content(text)
            prompt = """You are an expert in academic research summarization. Create an extremely concise TL;DR summary of the following academic abstract. Follow these guidelines:

1. Summarize the entire abstract in 3-4 short, clear sentences in English
2. Focus only on the most crucial information: main objective, key method, and primary finding or conclusion
3. Use simple, clear language while maintaining academic accuracy
4. Do not use separate headings or multiple paragraphs

Ensure the summary captures the essence of the research while being extremely concise."""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": preprocessed_text}
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error in generate_english_tldr: {e}")
            return "Unable to generate English summary."

    def translate_tldr_to_chinese(self, english_tldr):
        """將英文TL;DR翻譯成白話中文"""
        try:
            prompt = """你是專業的學術內容編輯，專門為網頁閱讀體驗優化學術摘要。請將以下英文學術摘要翻譯成適合網頁瀏覽的繁體中文：

格式要求：
• 控制在 80-120 字以內，方便手機閱讀
• 分成 2-3 個短句，每句用 "｜" 分隔
• 突出關鍵數據和結論

語言風格：
• 使用新聞式的客觀描述，避免過於口語
• 保留重要的專業術語，但加入簡單解釋
• 語調專業但親和，適合一般知識分子閱讀
• 強調實際影響和應用價值

範例格式：研究發現新藥X能降低50%的心臟病風險｜透過6個月臨床試驗證實｜預計明年進入第三期測試階段"""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": english_tldr}
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error in translate_tldr_to_chinese: {e}")
            return "無法翻譯摘要"

    def generate_tldr(self, text, target_language="zh-TW"):
        """使用兩步驟流程生成文章的TL;DR摘要"""
        try:
            # 第一步：生成英文摘要
            english_tldr = self.generate_english_tldr(text)
            
            # 第二步：翻譯成中文
            chinese_tldr = self.translate_tldr_to_chinese(english_tldr)
            
            return english_tldr, chinese_tldr
        except Exception as e:
            print(f"Error in generate_tldr: {e}")
            return "Unable to generate summary.", "無法生成摘要"

    # generate_keywords 方法已移除，不再需要

    def prepare_embedding_text(self, article, strategy="hybrid"):
        """
        準備用於嵌入的文本
        
        Args:
            article: 文章資料
            strategy: 嵌入策略 (hybrid, summary_only, original_only)
                
        Returns:
            str: 準備好的嵌入文本
        """
        if strategy == "summary_only":
            # 僅使用摘要資訊
            components = []
            if article.get('title'):
                components.append(f"Title: {article['title']}")
            if article.get('title_translated'):
                components.append(f"中文標題: {article['title_translated']}")
            if article.get('english_tldr'):
                components.append(f"Summary: {article['english_tldr']}")
            if article.get('chinese_tldr'):
                components.append(f"中文摘要: {article['chinese_tldr']}")
            return " | ".join(components)
            
        elif strategy == "original_only":
            # 使用清理後原文
            preprocessed_content = self.preprocess_content(article.get('full_content', ''))
            title_part = f"Title: {article.get('title', '')}"
            
            # 限制長度避免超過token限制
            max_content_length = 6000
            if len(preprocessed_content) > max_content_length:
                preprocessed_content = preprocessed_content[:max_content_length] + "..."
                
            return f"{title_part} | Content: {preprocessed_content}"
            
        else:  # hybrid
            # 混合策略：標題 + 原文摘要 + AI摘要
            components = []
            
            # 標題
            if article.get('title'):
                components.append(f"Title: {article['title']}")
            if article.get('title_translated'):
                components.append(f"中文標題: {article['title_translated']}")
                
            # 原文重點 (取前段)
            if article.get('full_content'):
                preprocessed_content = self.preprocess_content(article['full_content'])
                content_excerpt = preprocessed_content[:1500]
                if len(preprocessed_content) > 1500:
                    content_excerpt += "..."
                components.append(f"Original: {content_excerpt}")
                
            # AI摘要
            if article.get('english_tldr'):
                components.append(f"Summary: {article['english_tldr']}")
            if article.get('chinese_tldr'):
                components.append(f"中文摘要: {article['chinese_tldr']}")
                
            return " | ".join(components)

    def generate_embeddings(self, text_list):
        """
        生成文本的向量嵌入
        
        Args:
            text_list: 文本列表
            
        Returns:
            list: 向量列表
        """
        if not self.enable_embeddings or not text_list:
            return [None] * len(text_list)
            
        try:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=text_list
            )
            
            embeddings = [data.embedding for data in response.data]
            print(f"✅ 成功生成 {len(embeddings)} 個向量嵌入 (維度: {len(embeddings[0]) if embeddings else 0})")
            return embeddings
            
        except Exception as e:
            print(f"❌ 向量嵌入生成失敗: {e}")
            return [None] * len(text_list)

    def fetch_rss_basic(self, url):
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
            
            # 使用正則表達式從 content 中提取 DOI
            doi_match = re.search(r'DOI:\s*<a[^>]*>(.*?)</a>', content)
            doi = doi_match.group(1) if doi_match else None
            
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

    def load_existing_data_for_source(self, source):
        """從Supabase加載特定源的現有數據"""
        response = self.supabase.table("rss_entries").select("*").eq("source", source).execute()
        return response.data

    def save_rss_data(self, source, entries):
        """將RSS源的數據保存到Supabase（包含向量嵌入）"""
        for entry in entries:
            try:
                existing = self.supabase.table("rss_entries").select("*").eq("source", source).eq("pmid", entry['pmid']).execute()
                
                if existing.data:
                    # 對於已存在的條目，更新相關欄位
                    update_data = {
                        "doi": entry['doi']
                    }
                    
                    # 如果有新的嵌入資料，也更新
                    if entry.get('embedding') is not None:
                        update_data.update({
                            "english_tldr": entry.get('english_tldr', ''),
                            "embedding": entry['embedding'],
                            "embedding_text": entry.get('embedding_text', ''),
                            "embedding_strategy": self.embedding_strategy
                        })
                    
                    self.supabase.table("rss_entries").update(update_data).eq("source", source).eq("pmid", entry['pmid']).execute()
                    print(f"Updated entry {entry['pmid']} for source {source}")
                else:
                    # 對於新條目，插入所有字段
                    insert_data = {
                        "source": source,
                        "title": entry['title'],
                        "title_translated": entry.get('title_translated', ''),
                        "link": entry['link'],
                        "published": entry['published'],
                        "tldr": entry.get('chinese_tldr', ''),  # 保持原有欄位相容性
                        "english_tldr": entry.get('english_tldr', ''),
                        "pmid": entry['pmid'],
                        "doi": entry['doi'],
                        "embedding": entry.get('embedding'),
                        "embedding_text": entry.get('embedding_text', ''),
                        "embedding_strategy": self.embedding_strategy if entry.get('embedding') else None,
                        "likes_count": 0 
                    }
                    
                    self.supabase.table("rss_entries").insert(insert_data).execute()
                    print(f"Inserted new entry {entry['pmid']} for source {source}")
            except Exception as e:
                print(f"Error processing entry {entry['pmid']} for source {source}: {e}")
                print(f"Entry data: {entry}")

    def process_rss_sources(self, sources):
        """處理所有RSS來源並立即保存數據（包含向量嵌入）"""
        for name, url in sources.items():
            try:
                print(f"Processing source: {name}")
                new_feed_data = self.fetch_rss_basic(url)
                existing_entries = self.load_existing_data_for_source(name)
                existing_pmids = {entry['pmid']: entry for entry in existing_entries if 'pmid' in entry}
                
                new_entries = []
                updated_entries = []
                
                for entry in new_feed_data['entries']:
                    if entry['pmid'] not in existing_pmids:
                        # 處理新文章
                        print(f"  Processing new article: {entry['title'][:60]}...")
                        
                        # 翻譯標題
                        entry['title_translated'] = self.translate_title(entry['title'])
                        
                        # 生成摘要（兩步驟）
                        english_tldr, chinese_tldr = self.generate_tldr(entry['full_content'])
                        entry['english_tldr'] = english_tldr
                        entry['chinese_tldr'] = chinese_tldr
                        

                        
                        new_entries.append(entry)
                    else:
                        # 對於重複文章，只更新DOI
                        existing_entry = existing_pmids[entry['pmid']]
                        if existing_entry.get('doi') != entry['doi']:
                            existing_entry['doi'] = entry['doi']
                            updated_entries.append(existing_entry)
                
                # 批量生成向量嵌入（僅針對新文章）
                if new_entries and self.enable_embeddings:
                    print(f"  Generating embeddings for {len(new_entries)} new articles...")
                    embedding_texts = []
                    
                    for entry in new_entries:
                        embedding_text = self.prepare_embedding_text(entry, self.embedding_strategy)
                        entry['embedding_text'] = embedding_text
                        embedding_texts.append(embedding_text)
                    
                    embeddings = self.generate_embeddings(embedding_texts)
                    
                    # 將嵌入向量加入文章資料
                    for i, embedding in enumerate(embeddings):
                        new_entries[i]['embedding'] = embedding
                
                # 合併新文章和需要更新的文章
                entries_to_save = new_entries + updated_entries
                
                if entries_to_save:
                    self.save_rss_data(name, entries_to_save)
                    print(f"Processed {len(new_entries)} new entries and updated {len(updated_entries)} existing entries for {name}")
                else:
                    print(f"No new entries or updates for {name}")
            except Exception as e:
                print(f"Error processing source {name}: {e}")
                continue

    def load_rss_sources(self, file_path='rss_sources.json'):
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


def main():
    """主程序入口"""
    try:
        # 初始化處理器
        processor = LiteratureProcessor()
        
        # 載入RSS來源
        rss_sources = processor.load_rss_sources()
        
        # 處理所有RSS來源
        processor.process_rss_sources(rss_sources)
        print("RSS data processing completed successfully")
        
    except Exception as e:
        print(f"An error occurred during RSS processing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
