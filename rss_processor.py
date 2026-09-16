import feedparser
import json
import datetime
import time
import os
import sys
import re
import requests
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
from openai import OpenAI
from supabase import create_client, Client

class LiteratureProcessor:
    def __init__(self):
        """初始化文獻處理器"""
        # 初始化 OpenAI 客戶端
        self.api_key = self.get_openai_api_key()
        self.client = OpenAI(api_key=self.api_key)
        self.model = "gpt-5-mini"
        
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

    def fetch_pubmed_metadata(self, pmids):
        """透過 PubMed E-utilities 批次取得文章的研究類型 (Publication Type)
        與 PMC ID（有 PMC ID 代表 PubMed Central 有提供免費全文）。
        """
        pmids = [p for p in pmids if p]
        if not pmids:
            return {}

        try:
            response = requests.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
                params={
                    "db": "pubmed",
                    "id": ",".join(pmids),
                    "retmode": "xml",
                    "tool": "audslp_rss",
                },
                timeout=30
            )
            response.raise_for_status()
            root = ET.fromstring(response.content)

            result = {}
            for article in root.findall(".//PubmedArticle"):
                pmid_el = article.find(".//MedlineCitation/PMID")
                if pmid_el is None or not pmid_el.text:
                    continue
                pub_types = [
                    pt.text for pt in article.findall(".//PublicationTypeList/PublicationType")
                    if pt.text
                ]
                pmc_id = None
                for article_id in article.findall(".//ArticleIdList/ArticleId"):
                    if article_id.get("IdType") == "pmc":
                        pmc_id = article_id.text
                        break
                result[pmid_el.text] = {"publication_types": pub_types, "pmc_id": pmc_id}

            print(f"✅ 成功取得 {len(result)}/{len(pmids)} 篇文章的中繼資料")
            return result
        except Exception as e:
            print(f"❌ 取得文章中繼資料失敗: {e}")
            return {}

    def fetch_article_details_from_pubmed(self, pmid):
        """直接從 PubMed E-utilities 取得單篇文章的標題、摘要全文、doi、研究類型。

        跟 fetch_rss_basic 不同，這個方法不依賴 RSS feed（RSS 只保留最新 15 篇，
        舊文章可能已經不在裡面），適合用來重新處理已存在但資料有誤的文章。
        """
        try:
            response = requests.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
                params={
                    "db": "pubmed",
                    "id": pmid,
                    "retmode": "xml",
                    "tool": "audslp_rss",
                },
                timeout=30
            )
            response.raise_for_status()
            root = ET.fromstring(response.content)

            article = root.find(".//PubmedArticle")
            if article is None:
                print(f"❌ PubMed 找不到 pmid={pmid} 的文章")
                return None

            title_el = article.find(".//ArticleTitle")
            title = "".join(title_el.itertext()).strip() if title_el is not None else ""

            abstract_parts = []
            for ab_text in article.findall(".//Abstract/AbstractText"):
                label = ab_text.get("Label")
                text = "".join(ab_text.itertext()).strip()
                if not text:
                    continue
                abstract_parts.append(f"{label}: {text}" if label else text)
            full_content = "\n".join(abstract_parts)

            doi = None
            pmc_id = None
            for article_id in article.findall(".//ArticleIdList/ArticleId"):
                id_type = article_id.get("IdType")
                if id_type == "doi":
                    doi = article_id.text
                elif id_type == "pmc":
                    pmc_id = article_id.text

            pub_types = [
                pt.text for pt in article.findall(".//PublicationTypeList/PublicationType")
                if pt.text
            ]

            return {
                "title": title,
                "full_content": full_content,
                "doi": doi,
                "publication_types": pub_types,
                "pmc_id": pmc_id,
            }
        except Exception as e:
            print(f"❌ 取得文章詳細資料失敗 (pmid={pmid}): {e}")
            return None

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
                        "publication_types": entry.get('publication_types', []),
                        "pmc_id": entry.get('pmc_id'),
                        "likes_count": 0
                    }
                    
                    self.supabase.table("rss_entries").insert(insert_data).execute()
                    print(f"Inserted new entry {entry['pmid']} for source {source}")
            except Exception as e:
                print(f"Error processing entry {entry['pmid']} for source {source}: {e}")
                print(f"Entry data: {entry}")

    def reprocess_articles(self, pmids):
        """重新處理指定 pmid 的既有文章：重新翻譯標題、重新生成中英文摘要、
        重新取得研究類型與向量嵌入，並覆蓋 Supabase 裡同一篇文章的資料。

        用於資料有誤或當初 OpenAI 呼叫失敗時的補救，不會新增資料列，
        也不會動到 id/source/link/published/pmid/likes_count。
        """
        for pmid in pmids:
            try:
                print(f"重新處理 pmid={pmid} ...")
                existing = self.supabase.table("rss_entries").select("*").eq("pmid", pmid).execute()
                if not existing.data:
                    print(f"❌ 找不到 pmid={pmid} 的既有文章，略過（新文章請用一般流程）")
                    continue

                article_detail = self.fetch_article_details_from_pubmed(pmid)
                if not article_detail or not article_detail.get("full_content"):
                    print(f"❌ 無法從 PubMed 取得 pmid={pmid} 的完整資料，略過")
                    continue

                title_translated = self.translate_title(article_detail["title"])
                english_tldr, chinese_tldr = self.generate_tldr(article_detail["full_content"])

                embedding_source = {
                    "title": article_detail["title"],
                    "title_translated": title_translated,
                    "full_content": article_detail["full_content"],
                    "english_tldr": english_tldr,
                    "chinese_tldr": chinese_tldr,
                }
                embedding_text = self.prepare_embedding_text(embedding_source, self.embedding_strategy)
                embeddings = self.generate_embeddings([embedding_text]) if self.enable_embeddings else [None]
                embedding = embeddings[0] if embeddings else None

                update_data = {
                    "title": article_detail["title"],
                    "title_translated": title_translated,
                    "tldr": chinese_tldr,
                    "english_tldr": english_tldr,
                    "doi": article_detail.get("doi") or existing.data[0].get("doi"),
                    "publication_types": article_detail.get("publication_types", []),
                    "pmc_id": article_detail.get("pmc_id"),
                    "embedding": embedding,
                    "embedding_text": embedding_text,
                    "embedding_strategy": self.embedding_strategy if embedding else None,
                }

                self.supabase.table("rss_entries").update(update_data).eq("pmid", pmid).execute()
                print(f"✅ 已重新處理並更新 pmid={pmid}")
            except Exception as e:
                print(f"❌ 重新處理 pmid={pmid} 失敗: {e}")

    def process_rss_sources(self, sources, max_new_entries=None):
        """處理所有RSS來源並立即保存數據（包含向量嵌入）

        Args:
            max_new_entries: 測試模式用，限制本次執行最多處理幾篇「新」文章
                （已存在文章的 DOI 更新不受此限制）。None 代表不限制。
        """
        processed_new_count = 0
        for name, url in sources.items():
            if max_new_entries is not None and processed_new_count >= max_new_entries:
                print(f"已達測試上限 {max_new_entries} 篇新文章，停止處理後續來源")
                break
            try:
                print(f"Processing source: {name}")
                new_feed_data = self.fetch_rss_basic(url)
                existing_entries = self.load_existing_data_for_source(name)
                existing_pmids = {entry['pmid']: entry for entry in existing_entries if 'pmid' in entry}
                
                new_entries = []
                updated_entries = []
                
                for entry in new_feed_data['entries']:
                    if max_new_entries is not None and processed_new_count >= max_new_entries:
                        print(f"  已達測試上限 {max_new_entries} 篇新文章，停止處理來源 {name} 的後續項目")
                        break
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
                        processed_new_count += 1
                    else:
                        # 對於重複文章，只更新DOI
                        existing_entry = existing_pmids[entry['pmid']]
                        if existing_entry.get('doi') != entry['doi']:
                            existing_entry['doi'] = entry['doi']
                            updated_entries.append(existing_entry)
                
                # 批量取得研究類型與 PMC 免費全文資訊（僅針對新文章）
                if new_entries:
                    print(f"  Fetching PubMed metadata for {len(new_entries)} new articles...")
                    metadata_by_pmid = self.fetch_pubmed_metadata(
                        [entry['pmid'] for entry in new_entries]
                    )
                    for entry in new_entries:
                        meta = metadata_by_pmid.get(entry['pmid'], {})
                        entry['publication_types'] = meta.get('publication_types', [])
                        entry['pmc_id'] = meta.get('pmc_id')

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
        # 重新處理模式：只重跑指定 pmid 的既有文章，不做一般 RSS 掃描
        reprocess_pmids_env = os.environ.get("REPROCESS_PMIDS")
        if reprocess_pmids_env:
            pmids = [p.strip() for p in reprocess_pmids_env.split(",") if p.strip()]
            print(f"🔧 重新處理模式：{pmids}")
            processor = LiteratureProcessor()
            processor.reprocess_articles(pmids)
            print("Reprocessing completed successfully")
            return

        # 測試模式：限制本次最多處理幾篇新文章
        max_new_entries_env = os.environ.get("MAX_NEW_ENTRIES")
        max_new_entries = int(max_new_entries_env) if max_new_entries_env else None
        if max_new_entries is not None:
            print(f"⚠️ 測試模式：本次最多只處理 {max_new_entries} 篇新文章")

        # 初始化處理器
        processor = LiteratureProcessor()

        # 載入RSS來源
        rss_sources = processor.load_rss_sources()

        # 處理所有RSS來源
        processor.process_rss_sources(rss_sources, max_new_entries=max_new_entries)
        print("RSS data processing completed successfully")
        
    except Exception as e:
        print(f"An error occurred during RSS processing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
