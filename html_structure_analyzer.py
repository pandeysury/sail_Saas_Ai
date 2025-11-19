#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
html_structure_analyzer.py - Analyze Aspose HTML structure for parsing logic

Usage:
    python html_structure_analyzer.py --file "C:\sms\andriki\documents\5.06 Passage Planning.html"
    python html_structure_analyzer.py --file "path/to/file.html" --output analysis.txt
"""

import argparse
import re
from pathlib import Path
from collections import Counter, defaultdict
from bs4 import BeautifulSoup, Tag, NavigableString
import json


class HTMLStructureAnalyzer:
    """Analyze HTML structure to understand document organization."""
    
    def __init__(self, html_path: Path):
        self.html_path = html_path
        self.html_content = html_path.read_text(encoding='utf-8', errors='replace')
        self.soup = BeautifulSoup(self.html_content, 'html.parser')
        self.body = self.soup.find('body')
        
    def analyze_all(self):
        """Run all analysis methods and return comprehensive report."""
        print("=" * 80)
        print(f"HTML STRUCTURE ANALYSIS: {self.html_path.name}")
        print("=" * 80)
        print()
        
        report = {
            "file": str(self.html_path),
            "file_size": len(self.html_content),
            "toc_analysis": self.analyze_toc(),
            "heading_analysis": self.analyze_headings(),
            "anchor_analysis": self.analyze_anchors(),
            "list_analysis": self.analyze_lists(),
            "paragraph_analysis": self.analyze_paragraphs(),
            "table_analysis": self.analyze_tables(),
            "style_analysis": self.analyze_styles(),
            "structure_hierarchy": self.analyze_hierarchy(),
            "sample_sections": self.extract_sample_sections(),
        }
        
        return report
    
    def analyze_toc(self):
        """Analyze Table of Contents structure."""
        print("\n" + "=" * 80)
        print("1. TABLE OF CONTENTS (TOC) ANALYSIS")
        print("=" * 80)
        
        toc_info = {
            "toc_indicators_found": [],
            "toc_links": [],
            "toc_structure": None,
            "toc_location": None,
        }
        
        # Look for TOC indicators
        toc_indicators = ['table of contents', 'toc', 'contents']
        for indicator in toc_indicators:
            elements = self.soup.find_all(string=re.compile(indicator, re.I))
            if elements:
                toc_info["toc_indicators_found"].append({
                    "indicator": indicator,
                    "count": len(elements),
                    "locations": [str(elem.parent.name) for elem in elements[:3]]
                })
        
        # Find all _Toc links
        toc_links = []
        for a in self.soup.find_all('a', href=True):
            href = a.get('href', '')
            if re.search(r'#_Toc\d+', href, re.I):
                text = a.get_text(strip=True)
                toc_links.append({
                    "href": href,
                    "text": text,
                    "parent": a.parent.name if a.parent else None,
                })
        
        toc_info["toc_links"] = toc_links[:10]  # First 10
        toc_info["total_toc_links"] = len(toc_links)
        
        # Find TOC container
        toc_container = self.soup.find(attrs={'data-toc': True})
        if toc_container:
            toc_info["toc_structure"] = {
                "tag": toc_container.name,
                "attributes": dict(toc_container.attrs),
                "child_count": len(list(toc_container.children))
            }
        
        # Print results
        print(f"\n📋 TOC Indicators Found: {len(toc_info['toc_indicators_found'])}")
        for ind in toc_info['toc_indicators_found']:
            print(f"   - '{ind['indicator']}': {ind['count']} occurrences in {ind['locations']}")
        
        print(f"\n🔗 TOC Links (_Toc): {toc_info['total_toc_links']} total")
        print("\n   Sample TOC Links (first 10):")
        for i, link in enumerate(toc_links[:10], 1):
            print(f"   {i}. href='{link['href']}' text='{link['text'][:60]}' parent=<{link['parent']}>")
        
        if toc_container:
            print(f"\n📦 TOC Container: <{toc_info['toc_structure']['tag']} {toc_info['toc_structure']['attributes']}>")
        else:
            print("\n⚠️  No explicit TOC container found (data-toc attribute)")
        
        return toc_info
    
    def analyze_headings(self):
        """Analyze heading structure and IDs."""
        print("\n" + "=" * 80)
        print("2. HEADING STRUCTURE ANALYSIS")
        print("=" * 80)
        
        heading_info = {
            "standard_headings": [],
            "paragraph_headings": [],
            "heading_ids": [],
            "id_patterns": Counter(),
        }
        
        # Standard HTML headings (H1-H6)
        for level in range(1, 7):
            headings = self.soup.find_all(f'h{level}')
            for h in headings:
                heading_id = h.get('id', '')
                heading_text = h.get_text(strip=True)
                heading_info["standard_headings"].append({
                    "level": level,
                    "id": heading_id,
                    "text": heading_text[:100],
                    "has_id": bool(heading_id),
                    "id_type": self._classify_id(heading_id),
                })
                
                if heading_id:
                    heading_info["id_patterns"][self._classify_id(heading_id)] += 1
        
        # Paragraph-style headings (common in Aspose)
        for p in self.soup.find_all('p'):
            classes = p.get('class', [])
            class_str = ' '.join(classes) if isinstance(classes, list) else str(classes)
            
            # Check if paragraph has heading-like class
            if re.search(r'heading\s*\d', class_str, re.I):
                heading_id = p.get('id', '')
                heading_text = p.get_text(strip=True)
                heading_info["paragraph_headings"].append({
                    "class": class_str,
                    "id": heading_id,
                    "text": heading_text[:100],
                    "has_id": bool(heading_id),
                    "id_type": self._classify_id(heading_id),
                })
                
                if heading_id:
                    heading_info["id_patterns"][self._classify_id(heading_id)] += 1
        
        # Print results
        print(f"\n📊 Standard Headings (H1-H6): {len(heading_info['standard_headings'])}")
        heading_levels = Counter(h['level'] for h in heading_info['standard_headings'])
        for level, count in sorted(heading_levels.items()):
            print(f"   - H{level}: {count} headings")
        
        print(f"\n📊 Paragraph Headings (Aspose style): {len(heading_info['paragraph_headings'])}")
        
        print(f"\n🆔 Heading ID Patterns:")
        for id_type, count in heading_info["id_patterns"].most_common():
            print(f"   - {id_type}: {count} headings")
        
        print(f"\n📝 Sample Headings with IDs (first 10):")
        all_headings = heading_info['standard_headings'] + heading_info['paragraph_headings']
        for i, h in enumerate([h for h in all_headings if h['has_id']][:10], 1):
            level_str = f"H{h.get('level', 'P')}"
            print(f"   {i}. <{level_str} id='{h['id']}'> {h['text'][:60]}")
        
        return heading_info
    
    def analyze_anchors(self):
        """Analyze anchor tags and name attributes."""
        print("\n" + "=" * 80)
        print("3. ANCHOR TAG ANALYSIS")
        print("=" * 80)
        
        anchor_info = {
            "link_anchors": [],
            "name_anchors": [],
            "toc_anchors": [],
            "anchor_patterns": Counter(),
        }
        
        for a in self.soup.find_all('a'):
            href = a.get('href', '')
            name = a.get('name', '')
            text = a.get_text(strip=True)
            
            if href:
                anchor_info["link_anchors"].append({
                    "href": href,
                    "text": text[:60],
                    "is_toc": bool(re.search(r'#_Toc\d+', href)),
                })
                
                if re.search(r'#_Toc\d+', href):
                    anchor_info["toc_anchors"].append({
                        "href": href,
                        "text": text[:60],
                    })
            
            if name:
                anchor_info["name_anchors"].append({
                    "name": name,
                    "text": text[:60],
                    "parent": a.parent.name if a.parent else None,
                })
                anchor_info["anchor_patterns"][self._classify_id(name)] += 1
        
        # Print results
        print(f"\n🔗 Total Anchors: {len(self.soup.find_all('a'))}")
        print(f"   - With href: {len(anchor_info['link_anchors'])}")
        print(f"   - With name: {len(anchor_info['name_anchors'])}")
        print(f"   - TOC links (_Toc): {len(anchor_info['toc_anchors'])}")
        
        print(f"\n🏷️  Anchor Name Patterns:")
        for pattern, count in anchor_info["anchor_patterns"].most_common():
            print(f"   - {pattern}: {count} anchors")
        
        print(f"\n📝 Sample Name Anchors (first 10):")
        for i, anc in enumerate(anchor_info['name_anchors'][:10], 1):
            print(f"   {i}. <a name='{anc['name']}'> {anc['text'][:50]} (parent: <{anc['parent']}>)")
        
        return anchor_info
    
    def analyze_lists(self):
        """Analyze list structure (ul/ol/li)."""
        print("\n" + "=" * 80)
        print("4. LIST STRUCTURE ANALYSIS")
        print("=" * 80)
        
        list_info = {
            "unordered_lists": 0,
            "ordered_lists": 0,
            "list_items": 0,
            "nested_lists": 0,
            "list_samples": [],
        }
        
        ul_count = len(self.soup.find_all('ul'))
        ol_count = len(self.soup.find_all('ol'))
        li_count = len(self.soup.find_all('li'))
        
        list_info["unordered_lists"] = ul_count
        list_info["ordered_lists"] = ol_count
        list_info["list_items"] = li_count
        
        # Find nested lists
        for ul in self.soup.find_all('ul'):
            if ul.find('ul') or ul.find('ol'):
                list_info["nested_lists"] += 1
        
        # Sample lists
        for ul in self.soup.find_all('ul')[:3]:
            items = [li.get_text(strip=True)[:80] for li in ul.find_all('li', recursive=False)]
            list_info["list_samples"].append({
                "items": items,
                "item_count": len(items),
            })
        
        # Print results
        print(f"\n📋 Lists:")
        print(f"   - Unordered lists (<ul>): {ul_count}")
        print(f"   - Ordered lists (<ol>): {ol_count}")
        print(f"   - List items (<li>): {li_count}")
        print(f"   - Nested lists: {list_info['nested_lists']}")
        
        print(f"\n📝 Sample Lists (first 3):")
        for i, sample in enumerate(list_info['list_samples'], 1):
            print(f"\n   List {i} ({sample['item_count']} items):")
            for j, item in enumerate(sample['items'][:5], 1):
                print(f"      {j}. {item}")
        
        return list_info
    
    def analyze_paragraphs(self):
        """Analyze paragraph structure and styles."""
        print("\n" + "=" * 80)
        print("5. PARAGRAPH ANALYSIS")
        print("=" * 80)
        
        para_info = {
            "total_paragraphs": 0,
            "styled_paragraphs": 0,
            "bold_paragraphs": 0,
            "class_distribution": Counter(),
            "style_patterns": Counter(),
        }
        
        paragraphs = self.soup.find_all('p')
        para_info["total_paragraphs"] = len(paragraphs)
        
        for p in paragraphs:
            # Classes
            classes = p.get('class', [])
            if classes:
                class_str = ' '.join(classes) if isinstance(classes, list) else str(classes)
                para_info["class_distribution"][class_str] += 1
            
            # Styles
            style = p.get('style', '')
            if style:
                para_info["styled_paragraphs"] += 1
                # Extract font-weight
                if 'font-weight' in style:
                    match = re.search(r'font-weight:\s*(\w+)', style)
                    if match and match.group(1) in ['bold', '700', '800', '900']:
                        para_info["bold_paragraphs"] += 1
                        para_info["style_patterns"]["bold"] += 1
        
        # Print results
        print(f"\n📄 Paragraphs: {para_info['total_paragraphs']}")
        print(f"   - With styles: {para_info['styled_paragraphs']}")
        print(f"   - Bold paragraphs: {para_info['bold_paragraphs']}")
        
        print(f"\n🎨 Top Paragraph Classes:")
        for cls, count in para_info["class_distribution"].most_common(10):
            print(f"   - '{cls}': {count} paragraphs")
        
        return para_info
    
    def analyze_tables(self):
        """Analyze table structure."""
        print("\n" + "=" * 80)
        print("6. TABLE ANALYSIS")
        print("=" * 80)
        
        table_info = {
            "total_tables": 0,
            "tables_with_toc": 0,
            "sample_tables": [],
        }
        
        tables = self.soup.find_all('table')
        table_info["total_tables"] = len(tables)
        
        for table in tables:
            text = table.get_text(" ", strip=True).lower()
            if 'table of contents' in text or 'toc' in text:
                table_info["tables_with_toc"] += 1
        
        # Sample first table
        if tables:
            first_table = tables[0]
            rows = first_table.find_all('tr')
            table_info["sample_tables"].append({
                "rows": len(rows),
                "cells": len(first_table.find_all(['td', 'th'])),
                "preview": first_table.get_text(" ", strip=True)[:200],
            })
        
        # Print results
        print(f"\n📊 Tables: {table_info['total_tables']}")
        print(f"   - Tables with TOC content: {table_info['tables_with_toc']}")
        
        if table_info["sample_tables"]:
            sample = table_info["sample_tables"][0]
            print(f"\n📝 First Table:")
            print(f"   - Rows: {sample['rows']}")
            print(f"   - Cells: {sample['cells']}")
            print(f"   - Preview: {sample['preview']}")
        
        return table_info
    
    def analyze_styles(self):
        """Analyze inline styles and CSS."""
        print("\n" + "=" * 80)
        print("7. STYLE & CSS ANALYSIS")
        print("=" * 80)
        
        style_info = {
            "elements_with_style": 0,
            "common_style_properties": Counter(),
            "aspose_attributes": Counter(),
        }
        
        for elem in self.soup.find_all(True):
            # Inline styles
            if elem.get('style'):
                style_info["elements_with_style"] += 1
                style = elem.get('style', '')
                # Extract properties
                for prop in re.findall(r'([\w-]+)\s*:', style):
                    style_info["common_style_properties"][prop] += 1
            
            # Aspose-specific attributes
            for attr in elem.attrs:
                if attr.startswith('-aw-'):
                    style_info["aspose_attributes"][attr] += 1
        
        # Print results
        print(f"\n🎨 Styled Elements: {style_info['elements_with_style']}")
        
        print(f"\n📊 Common Style Properties (top 10):")
        for prop, count in style_info["common_style_properties"].most_common(10):
            print(f"   - {prop}: {count} occurrences")
        
        print(f"\n🔧 Aspose Attributes:")
        for attr, count in style_info["aspose_attributes"].most_common(10):
            print(f"   - {attr}: {count} occurrences")
        
        return style_info
    
    def analyze_hierarchy(self):
        """Analyze document hierarchy and nesting."""
        print("\n" + "=" * 80)
        print("8. DOCUMENT HIERARCHY")
        print("=" * 80)
        
        hierarchy_info = {
            "max_depth": 0,
            "structure_outline": [],
        }
        
        def get_depth(element):
            depth = 0
            parent = element.parent
            while parent and parent.name != 'body':
                depth += 1
                parent = parent.parent
            return depth
        
        # Find max depth
        max_depth = 0
        for elem in self.body.find_all(True):
            depth = get_depth(elem)
            max_depth = max(max_depth, depth)
        
        hierarchy_info["max_depth"] = max_depth
        
        # Create structure outline (first 20 significant elements)
        significant_tags = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'ul', 'ol', 'table']
        count = 0
        for elem in self.body.find_all(significant_tags):
            if count >= 20:
                break
            depth = get_depth(elem)
            text = elem.get_text(strip=True)[:50]
            elem_id = elem.get('id', '')
            hierarchy_info["structure_outline"].append({
                "tag": elem.name,
                "depth": depth,
                "id": elem_id,
                "text": text,
            })
            count += 1
        
        # Print results
        print(f"\n📏 Max Nesting Depth: {max_depth}")
        
        print(f"\n🌳 Document Structure Outline (first 20 significant elements):")
        for i, item in enumerate(hierarchy_info['structure_outline'], 1):
            indent = "  " * item['depth']
            id_str = f" id='{item['id']}'" if item['id'] else ""
            print(f"   {i}. {indent}<{item['tag']}{id_str}> {item['text']}")
        
        return hierarchy_info
    
    def extract_sample_sections(self):
        """Extract sample sections with their complete HTML."""
        print("\n" + "=" * 80)
        print("9. SAMPLE SECTIONS (RAW HTML)")
        print("=" * 80)
        
        samples = []
        
        # Find first heading with TOC ID
        heading_with_toc = None
        for tag in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            for h in self.soup.find_all(tag):
                elem_id = h.get('id', '')
                if '_Toc' in elem_id:
                    heading_with_toc = h
                    break
            if heading_with_toc:
                break
        
        if heading_with_toc:
            # Get heading + next 3 siblings
            section_html = str(heading_with_toc)
            sibling = heading_with_toc.next_sibling
            sibling_count = 0
            while sibling and sibling_count < 3:
                if isinstance(sibling, Tag):
                    section_html += "\n" + str(sibling)
                    sibling_count += 1
                sibling = sibling.next_sibling
            
            samples.append({
                "description": "Heading with TOC ID + following content",
                "html": section_html[:1000],  # First 1000 chars
            })
            
            print("\n📄 Sample Section: Heading with TOC ID")
            print("-" * 80)
            print(section_html[:1000])
            print("..." if len(section_html) > 1000 else "")
            print("-" * 80)
        
        # Find a list
        ul = self.soup.find('ul')
        if ul:
            list_html = str(ul)[:800]
            samples.append({
                "description": "Sample list structure",
                "html": list_html,
            })
            
            print("\n📋 Sample List Structure:")
            print("-" * 80)
            print(list_html)
            print("..." if len(str(ul)) > 800 else "")
            print("-" * 80)
        
        return samples
    
    def _classify_id(self, id_str):
        """Classify ID type based on pattern."""
        if not id_str:
            return "no_id"
        if re.match(r'^_Toc\d+$', id_str):
            return "_Toc (TOC)"
        if re.match(r'^sec-[\w-]+$', id_str):
            return "sec- (generated)"
        if re.match(r'^[\w-]+$', id_str):
            return "custom"
        return "other"
    
    def save_report(self, output_path: Path, report: dict):
        """Save analysis report to file."""
        with output_path.open('w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(f"HTML STRUCTURE ANALYSIS REPORT\n")
            f.write(f"File: {report['file']}\n")
            f.write(f"File Size: {report['file_size']:,} bytes\n")
            f.write("=" * 80 + "\n\n")
            
            # Write each section
            sections = [
                ("toc_analysis", "TABLE OF CONTENTS ANALYSIS"),
                ("heading_analysis", "HEADING STRUCTURE ANALYSIS"),
                ("anchor_analysis", "ANCHOR TAG ANALYSIS"),
                ("list_analysis", "LIST STRUCTURE ANALYSIS"),
                ("paragraph_analysis", "PARAGRAPH ANALYSIS"),
                ("table_analysis", "TABLE ANALYSIS"),
                ("style_analysis", "STYLE & CSS ANALYSIS"),
                ("structure_hierarchy", "DOCUMENT HIERARCHY"),
            ]
            
            for key, title in sections:
                f.write(f"\n{'=' * 80}\n")
                f.write(f"{title}\n")
                f.write(f"{'=' * 80}\n\n")
                f.write(json.dumps(report[key], indent=2, ensure_ascii=False))
                f.write("\n\n")
            
            # Sample sections
            f.write(f"\n{'=' * 80}\n")
            f.write("SAMPLE HTML SECTIONS\n")
            f.write(f"{'=' * 80}\n\n")
            for sample in report['sample_sections']:
                f.write(f"\n{sample['description']}:\n")
                f.write("-" * 80 + "\n")
                f.write(sample['html'])
                f.write("\n" + "-" * 80 + "\n\n")
        
        print(f"\n✅ Full report saved to: {output_path}")


def extract_parsing_recommendations(report: dict):
    """Generate parsing recommendations based on analysis."""
    print("\n" + "=" * 80)
    print("10. PARSING RECOMMENDATIONS")
    print("=" * 80)
    
    recommendations = []
    
    # TOC extraction
    toc_count = report['toc_analysis']['total_toc_links']
    if toc_count > 0:
        recommendations.append({
            "priority": "HIGH",
            "component": "TOC Extraction",
            "recommendation": f"Found {toc_count} TOC links. Use regex to extract BEFORE any HTML cleaning.",
            "code_hint": "regex: r'<a[^>]+href=\"#(_Toc[0-9]+)\"[^>]*>(.*?)</a>'"
        })
    
    # Heading ID patterns
    id_patterns = report['heading_analysis']['id_patterns']
    if '_Toc (TOC)' in id_patterns:
        recommendations.append({
            "priority": "HIGH",
            "component": "Heading Matching",
            "recommendation": f"Found {id_patterns['_Toc (TOC)']} headings with _Toc IDs. Normalize text by removing section numbers (e.g., '4.3.5').",
            "code_hint": "text = re.sub(r'^\\s*[\\d\\.]+\\s+', '', text)"
        })
    
    # Paragraph headings
    para_headings = len(report['heading_analysis']['paragraph_headings'])
    if para_headings > 0:
        recommendations.append({
            "priority": "MEDIUM",
            "component": "Heading Detection",
            "recommendation": f"Found {para_headings} paragraph-style headings. Check for bold paragraphs and Heading classes.",
            "code_hint": "Check: font-weight:bold, class='Heading1', class='Heading2', etc."
        })
    
    # Lists
    list_count = report['list_analysis']['unordered_lists'] + report['list_analysis']['ordered_lists']
    if list_count > 0:
        recommendations.append({
            "priority": "HIGH",
            "component": "List Merging",
            "recommendation": f"Found {list_count} lists. Implement merge_heading_with_content() to keep lists with headings.",
            "code_hint": "Look ahead 20 nodes after heading and merge <li> elements"
        })
    
    # Aspose attributes
    aspose_attrs = report['style_analysis']['aspose_attributes']
    if aspose_attrs:
        recommendations.append({
            "priority": "LOW",
            "component": "Cleanup",
            "recommendation": f"Found {len(aspose_attrs)} Aspose-specific attributes. Clean these after TOC extraction.",
            "code_hint": "Remove attributes starting with '-aw-'"
        })
    
    # Print recommendations
    print("\n🎯 Recommended Parsing Strategy:\n")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. [{rec['priority']}] {rec['component']}")
        print(f"   ➜ {rec['recommendation']}")
        print(f"   💡 Code hint: {rec['code_hint']}")
        print()
    
    return recommendations


def generate_sample_code(report: dict):
    """Generate sample parsing code based on analysis."""
    print("\n" + "=" * 80)
    print("11. SAMPLE PARSING CODE")
    print("=" * 80)
    
    code = """
# Sample parsing code based on your HTML structure

import re
from bs4 import BeautifulSoup

def parse_aspose_html(html_content: str):
    \"\"\"Parse Aspose HTML with TOC extraction.\"\"\"
    
    # STEP 1: Extract TOC map BEFORE any cleaning
    toc_map = extract_toc_map(html_content)
    print(f"Extracted {len(toc_map)} TOC mappings")
    
    # STEP 2: Parse HTML
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # STEP 3: Find all headings (including paragraph-style)
    headings = []
    
    # Standard headings
    for tag in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
        for h in soup.find_all(tag):
            headings.append({
                'tag': tag,
                'id': h.get('id', ''),
                'text': h.get_text(strip=True),
                'element': h
            })
    
    # Paragraph headings (Aspose style)
    for p in soup.find_all('p'):
        classes = ' '.join(p.get('class', []))
        if re.search(r'Heading\\d', classes, re.I):
            headings.append({
                'tag': 'p',
                'id': p.get('id', ''),
                'text': p.get_text(strip=True),
                'element': p
            })
    
    # STEP 4: Match headings to TOC IDs
    for heading in headings:
        text = heading['text']
        normalized = normalize_text(text)
        
        if normalized in toc_map:
            heading['toc_id'] = toc_map[normalized]
            print(f"✅ Matched: '{text}' → {toc_map[normalized]}")
        else:
            heading['toc_id'] = heading['id'] or generate_fallback_id(text)
            print(f"⚠️  No TOC match for: '{text}', using fallback")
    
    return headings


def extract_toc_map(html: str) -> dict:
    \"\"\"Extract TOC mappings from HTML.\"\"\"
    toc_map = {}
    
    # Find all <a href="#_Toc...">Text</a>
    pattern = r'<a[^>]+href=\"#(_Toc[0-9]+)\"[^>]*>(.*?)</a>'
    
    for match in re.finditer(pattern, html, re.I | re.S):
        toc_id = match.group(1)
        text = strip_html_tags(match.group(2))
        
        if text and toc_id:
            normalized = normalize_text(text)
            toc_map[normalized] = toc_id
    
    return toc_map


def normalize_text(text: str) -> str:
    \"\"\"Normalize text for matching.\"\"\"
    # Remove leading section numbers
    text = re.sub(r'^\\s*[\\d\\.]+\\s+', '', text)
    
    # Lowercase and normalize whitespace
    text = text.lower().strip()
    text = re.sub(r'\\s+', ' ', text)
    
    return text


def strip_html_tags(html: str) -> str:
    \"\"\"Remove HTML tags from text.\"\"\"
    # Remove script/style content
    html = re.sub(r'<script[^>]*>.*?</script>', ' ', html, flags=re.I|re.S)
    html = re.sub(r'<style[^>]*>.*?</style>', ' ', html, flags=re.I|re.S)
    
    # Remove tags
    html = re.sub(r'<[^>]+>', ' ', html)
    
    # Decode HTML entities
    import html as html_module
    html = html_module.unescape(html)
    
    # Normalize whitespace
    html = re.sub(r'\\s+', ' ', html).strip()
    
    return html


def generate_fallback_id(text: str) -> str:
    \"\"\"Generate fallback ID if no TOC match.\"\"\"
    import hashlib
    
    # Create slug from text
    slug = re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')
    
    if not slug:
        # Use hash if text is empty or all non-alphanumeric
        slug = hashlib.sha1(text.encode('utf-8')).hexdigest()[:10]
    
    return f'sec-{slug[:120]}'


# Example usage:
if __name__ == '__main__':
    with open('5.06 Passage Planning.html', 'r', encoding='utf-8') as f:
        html = f.read()
    
    headings = parse_aspose_html(html)
    
    print(f"\\nFound {len(headings)} headings:")
    for h in headings[:10]:
        print(f"  - {h['text'][:60]} → {h['toc_id']}")
"""
    
    print(code)
    
    return code


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Aspose HTML structure for parsing logic development"
    )
    parser.add_argument(
        '--file', 
        required=True, 
        help='Path to HTML file to analyze'
    )
    parser.add_argument(
        '--output',
        default='html_analysis_report.txt',
        help='Output file for detailed report (default: html_analysis_report.txt)'
    )
    parser.add_argument(
        '--generate-code',
        action='store_true',
        help='Generate sample parsing code'
    )
    
    args = parser.parse_args()
    
    html_path = Path(args.file)
    if not html_path.exists():
        print(f"❌ Error: File not found: {html_path}")
        return 1
    
    # Run analysis
    analyzer = HTMLStructureAnalyzer(html_path)
    report = analyzer.analyze_all()
    
    # Generate recommendations
    recommendations = extract_parsing_recommendations(report)
    
    # Generate sample code if requested
    if args.generate_code:
        sample_code = generate_sample_code(report)
        
        # Save sample code
        code_path = Path('sample_parser.py')
        code_path.write_text(sample_code, encoding='utf-8')
        print(f"\n✅ Sample parsing code saved to: {code_path}")
    
    # Save detailed report
    output_path = Path(args.output)
    analyzer.save_report(output_path, report)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"\n📊 Summary:")
    print(f"   - TOC Links: {report['toc_analysis']['total_toc_links']}")
    print(f"   - Headings: {len(report['heading_analysis']['standard_headings']) + len(report['heading_analysis']['paragraph_headings'])}")
    print(f"   - Lists: {report['list_analysis']['unordered_lists'] + report['list_analysis']['ordered_lists']}")
    print(f"   - Tables: {report['table_analysis']['total_tables']}")
    print(f"   - Max Nesting Depth: {report['structure_hierarchy']['max_depth']}")
    print(f"\n📄 Detailed report: {output_path}")
    
    if args.generate_code:
        print(f"🔧 Sample parser: sample_parser.py")
    
    print("\n💡 Next Steps:")
    print("   1. Review the detailed report to understand HTML structure")
    print("   2. Check parsing recommendations above")
    print("   3. Use --generate-code flag to create sample parser")
    print("   4. Update your indexer with appropriate TOC extraction logic")
    
    return 0


if __name__ == '__main__':
    exit(main())