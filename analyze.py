import os
import re
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from datetime import datetime
from collections import Counter, defaultdict
from tqdm import tqdm
import openai
from openai import OpenAI
import asyncio
import time
import aiohttp
import backoff
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import numpy as np
from wordcloud import WordCloud

# Load environment variables
load_dotenv()

# Configure OpenAI API key and create client
openai.api_key = "YOUR_API_KEY"
client = OpenAI(api_key=openai.api_key)

class ComprehensiveMeetingAnalyzer:
    def __init__(self, folder_path):
        """Initialize the comprehensive meeting analyzer."""
        self.folder_path = folder_path
        self.transcripts = {}
        
        # Settings for handling large files
        self.chunk_size = 80000  # Conservative chunk size
        self.max_api_retries = 5
        
        # Analysis results storage
        self.meeting_analyses = {}
        self.global_patterns = {}
        self.speaker_network = {}
        self.topic_network = {}
        self.temporal_trends = {}
        self.anomalies = []
        
        # Create necessary directories
        os.makedirs('results', exist_ok=True)
        os.makedirs('intermediate_results', exist_ok=True)
        os.makedirs('visualizations', exist_ok=True)
        os.makedirs('reports', exist_ok=True)

    def parse_date_from_filename(self, filename):
        """Extract date from filename patterns."""
        if match := re.search(r'(\d{1,2})⧸(\d{1,2})⧸(\d{4})', filename):
            month, day, year = match.groups()
            return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
        elif match := re.search(r'(\d{8})-', filename):
            date_str = match.group(1)
            year = date_str[:4]
            month = date_str[4:6]
            day = date_str[6:8]
            return f"{year}-{month}-{day}"
        else:
            return "Unknown"

    def load_transcripts(self):
        """Load all transcript files from the folder."""
        print("Loading transcript files...")
        files = [f for f in os.listdir(self.folder_path) if f.endswith('.txt')]
        
        for file in tqdm(files):
            try:
                file_path = os.path.join(self.folder_path, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                date = self.parse_date_from_filename(file)
                year = date.split('-')[0] if date != "Unknown" else "Unknown"
                month = date.split('-')[1] if date != "Unknown" else "Unknown"
                
                self.transcripts[file] = {
                    'content': content,
                    'date': date,
                    'year': year,
                    'month': month,
                    'filename': file,
                    'file_size': len(content)
                }
            except Exception as e:
                print(f"Error loading {file}: {str(e)}")
        
        print(f"Loaded {len(self.transcripts)} transcript files.")
        
        # Sort transcripts by date
        self.transcripts = dict(sorted(self.transcripts.items(), 
                                      key=lambda x: x[1]['date'] if x[1]['date'] != "Unknown" else "9999-99-99"))
        
        # Log file size statistics
        sizes = [data['file_size'] for data in self.transcripts.values()]
        print(f"File size statistics: Min={min(sizes)}, Max={max(sizes)}, Avg={sum(sizes)/len(sizes):.0f} characters")
        print(f"Files larger than 500KB: {sum(1 for size in sizes if size > 500000)}")
        print(f"Files larger than 1MB: {sum(1 for size in sizes if size > 1000000)}")

    def smart_chunk_text(self, content):
        """Split content into chunks, trying to break at natural boundaries."""
        chunks = []
        current_position = 0
        content_length = len(content)
        
        while current_position < content_length:
            # Calculate end position for this chunk
            end_position = min(current_position + self.chunk_size, content_length)
            
            # If we're not at the end of the content, try to find a natural break
            if end_position < content_length:
                # Look for paragraph breaks or speaker transitions
                paragraph_break = content.rfind('\n\n', current_position, end_position)
                speaker_break = content.rfind('--------------------------------------------------', 
                                            current_position, end_position)
                
                # Find the best break point
                if speaker_break > current_position and speaker_break > paragraph_break:
                    end_position = speaker_break
                elif paragraph_break > current_position:
                    end_position = paragraph_break
            
            # Extract the chunk
            chunk = content[current_position:end_position]
            chunks.append(chunk)
            
            # Move to next position
            current_position = end_position
        
        return chunks

    def extract_speaker_segments(self, content):
        """Extract speaker segments from transcript content."""
        segments = []
        for chunk in content.split('--------------------------------------------------'):
            if 'Start:' in chunk and 'End:' in chunk and 'Duration:' in chunk:
                try:
                    lines = chunk.strip().split('\n')
                    speaker_line = lines[0]
                    speaker = speaker_line.split(':')[0].strip()
                    
                    # Extract time information
                    start_time = re.search(r'Start: (\d+:\d+:\d+\.\d+)', chunk)
                    end_time = re.search(r'End: (\d+:\d+:\d+\.\d+)', chunk)
                    duration = re.search(r'Duration: (\d+\.\d+)', chunk)
                    
                    # Extract the text
                    text_match = re.search(r'Text: (.*?)$', chunk, re.DOTALL)
                    text = text_match.group(1).strip() if text_match else ""
                    
                    segments.append({
                        'speaker': speaker,
                        'start': start_time.group(1) if start_time else "",
                        'end': end_time.group(1) if end_time else "",
                        'duration': float(duration.group(1)) if duration else 0.0,
                        'text': text
                    })
                except Exception as e:
                    continue  # Skip malformed segments
        
        return segments

    @backoff.on_exception(backoff.expo, 
                         (openai.APIError, openai.RateLimitError, aiohttp.ClientError),
                         max_tries=5, 
                         max_time=300)
    async def call_openai_agent(self, agent_request):
        """Call OpenAI API with retry logic."""
        messages = [{"role": "user", "content": agent_request["input"]}]
        
        # Run the synchronous client call in a separate thread to not block the event loop
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, 
            lambda: client.chat.completions.create(
                model="gpt-4",
                messages=messages
            )
        )
        
        # Create a response object with a result attribute to maintain compatibility
        class ResponseWrapper:
            def __init__(self, content):
                self.result = content
        
        return ResponseWrapper(response.choices[0].message.content)

    async def analyze_transcript(self, filename, transcript_data):
        """Perform comprehensive analysis of a transcript using multiple agent calls."""
        content = transcript_data['content']
        date = transcript_data['date']
        file_size = transcript_data['file_size']
        
        print(f"Processing {filename} ({file_size} characters, date: {date})")
        
        # Initialize result structure
        analysis_result = {
            'filename': filename,
            'date': date,
            'file_size': file_size,
            'open_ended_analysis': {},
            'directed_analysis': {},
            'speaker_analysis': {},
            'emergent_patterns': {},
            'error': None
        }
        
        # Process in chunks for large files
        if file_size > self.chunk_size:
            print(f"  Transcript is large ({file_size} chars), analyzing in chunks...")
            chunks = self.smart_chunk_text(content)
            print(f"  Split into {len(chunks)} chunks")
            chunk_results = []
            
            for i, chunk in enumerate(chunks):
                print(f"  Processing chunk {i+1}/{len(chunks)} ({len(chunk)} chars)...")
                try:
                    chunk_result = await self._analyze_chunk(chunk, f"{filename} (part {i+1}/{len(chunks)})")
                    chunk_results.append(chunk_result)
                    
                    # Save intermediate chunk results
                    os.makedirs('intermediate_results', exist_ok=True)
                    safe_filename = filename.replace("/", "_").replace("\\", "_")
                    chunk_file = f'intermediate_results/{safe_filename}_chunk_{i+1}.json'
                    with open(chunk_file, 'w') as f:
                        json.dump(chunk_result, f, indent=2)
                    
                    # Add delay between chunks
                    await asyncio.sleep(2)
                except Exception as e:
                    print(f"  Error processing chunk {i+1}: {str(e)}")
                    chunk_results.append({
                        "error": str(e)
                    })
            
            # Combine chunk results
            combined_result = self._combine_chunk_results(chunk_results)
            for key in combined_result:
                analysis_result[key] = combined_result[key]
        else:
            # Process entire transcript at once
            try:
                single_result = await self._analyze_chunk(content, filename)
                for key in single_result:
                    analysis_result[key] = single_result[key]
            except Exception as e:
                print(f"Error analyzing {filename}: {str(e)}")
                analysis_result['error'] = str(e)
        
        return analysis_result

    async def _analyze_chunk(self, content, chunk_name):
        """Perform multi-faceted analysis on a single chunk of transcript."""
        # Extract speaker segments
        segments = self.extract_speaker_segments(content)
        
        # Initialize result
        result = {
            'open_ended_analysis': {},
            'directed_analysis': {},
            'speaker_analysis': {},
            'emergent_patterns': {}
        }
        
        # If no segments found, use a fallback approach
        if not segments:
            print(f"  No segments found in {chunk_name}, using raw text...")
            all_text = content
        else:
            # Calculate speaker statistics
            speakers = {}
            for segment in segments:
                speaker = segment['speaker']
                duration = segment['duration']
                text = segment['text']
                
                if speaker not in speakers:
                    speakers[speaker] = {
                        'total_time': 0,
                        'segments': []
                    }
                
                speakers[speaker]['total_time'] += duration
                speakers[speaker]['segments'].append({
                    'duration': duration,
                    'text': text
                })
            
            # Prepare text for analysis
            all_text = " ".join([segment['text'] for segment in segments])
            
            # Store speaker data
            result['speaker_analysis']['speakers'] = {
                'count': len(speakers),
                'speaking_time': {speaker: data['total_time'] for speaker, data in speakers.items()}
            }
        
        # If all_text is still too large, create a representative sample
        if len(all_text) > self.chunk_size:
            if segments:
                # Take segments from beginning, middle and end
                sample_segments = []
                total_segments = len(segments)
                
                # Take segments from the beginning
                sample_segments.extend(segments[:min(20, total_segments // 3)])
                
                # Take segments from the middle
                mid_start = max(20, total_segments // 3)
                mid_end = min(total_segments - 20, 2 * total_segments // 3)
                sample_segments.extend(segments[mid_start:mid_end])
                
                # Take segments from the end
                sample_segments.extend(segments[-min(20, total_segments // 3):])
                
                # Create sample text
                sample_text = " ".join([segment['text'] for segment in sample_segments])
                print(f"  Created representative sample of {len(sample_text)} chars from {len(all_text)} chars total")
                all_text = sample_text
            else:
                # Sample from raw text
                text_parts = [
                    all_text[:min(self.chunk_size//3, len(all_text)//3)],
                    all_text[len(all_text)//2 - self.chunk_size//6:len(all_text)//2 + self.chunk_size//6],
                    all_text[-min(self.chunk_size//3, len(all_text)//3):]
                ]
                all_text = "\n\n[...]\n\n".join(text_parts)
        
        # 1. Open-ended analysis - discover what's in the transcript without guiding the AI
        try:
            open_ended_prompt = {
                "input": f"""Analyze this Manhattan, Kansas city council meeting transcript with an open mind. 
                
                Don't look for specific topics or patterns - just tell me what stands out to you most about this meeting.
                What were the most significant discussions? What seemed to be important to the speakers?
                
                Transcript:
                {all_text[:self.chunk_size]}
                
                Provide your analysis in JSON format with the following structure:
                {{
                    "primary_themes": [
                        {{
                            "theme": "Name of theme",
                            "description": "Brief description",
                            "notable_points": ["Point 1", "Point 2"]
                        }}
                    ],
                    "significant_discussions": [
                        {{
                            "topic": "Topic description",
                            "significance": "Why this matters",
                            "stakeholders": ["Groups or individuals with interest in this topic"]
                        }}
                    ],
                    "unexpected_elements": [
                        {{
                            "element": "Something surprising or unusual",
                            "context": "Why this is noteworthy"
                        }}
                    ]
                }}""",
                "agent": "discovery_agent",
                "stream": False
            }
            
            open_ended_response = await self.call_openai_agent(open_ended_prompt)
            
            try:
                result['open_ended_analysis'] = json.loads(open_ended_response.result)
            except:
                result['open_ended_analysis'] = {"error": "Failed to parse response", "raw": open_ended_response.result}
        
        except Exception as e:
            print(f"  Error in open-ended analysis: {str(e)}")
            result['open_ended_analysis'] = {"error": str(e)}
        
        # 2. Directed analysis - focus on Dr. Lind's specific interests
        try:
            directed_prompt = {
                "input": f"""Analyze this Manhattan, Kansas city council meeting transcript focusing on Dr. Lind's interests.
                
                Transcript:
                {all_text[:self.chunk_size]}
                
                Analyze for:
                1. Land use discussions (zoning, housing, development, urban planning)
                2. Economic development discussions (business growth, industry, employment, downtown development)
                3. Tax and budget discussions (revenue generation, property taxes, sales taxes, spending priorities)
                
                Don't just look for keywords - understand the context and content of discussions.
                
                Also identify any additional significant topics that appear important in this meeting.
                
                Provide your analysis in JSON format:
                {{
                    "land_use": {{
                        "present": true/false,
                        "significance": "High/Medium/Low",
                        "context": "How land use was discussed in this meeting",
                        "notable_points": ["Point 1", "Point 2"]
                    }},
                    "economic_development": {{
                        "present": true/false,
                        "significance": "High/Medium/Low",
                        "context": "How economic development was discussed",
                        "notable_points": ["Point 1", "Point 2"]
                    }},
                    "taxes_budget": {{
                        "present": true/false,
                        "significance": "High/Medium/Low",
                        "context": "How taxes/budget were discussed",
                        "notable_points": ["Point 1", "Point 2"]
                    }},
                    "other_significant_topics": [
                        {{
                            "topic": "Topic name",
                            "significance": "High/Medium/Low",
                            "context": "Brief context"
                        }}
                    ]
                }}""",
                "agent": "directed_analysis_agent",
                "stream": False
            }
            
            directed_response = await self.call_openai_agent(directed_prompt)
            
            try:
                result['directed_analysis'] = json.loads(directed_response.result)
            except:
                result['directed_analysis'] = {"error": "Failed to parse response", "raw": directed_response.result}
        
        except Exception as e:
            print(f"  Error in directed analysis: {str(e)}")
            result['directed_analysis'] = {"error": str(e)}
        
        # 3. Speaker analysis - understand participation patterns
        if segments and len(segments) > 0:
            try:
                speaker_prompt = {
                    "input": f"""Analyze the speaker dynamics in this Manhattan, Kansas city council meeting.
                    
                    Speaker data:
                    {json.dumps([{"speaker": speaker, "speaking_time": data['total_time'], "segments": len(data['segments'])} 
                                for speaker, data in speakers.items()])}
                    
                    Based on this data and these excerpts from the meeting:
                    {all_text[:min(20000, len(all_text))]}
                    
                    Please analyze:
                    1. Power dynamics - who seems to be leading or dominating discussions?
                    2. Interaction patterns - how do speakers interact with each other?
                    3. Topic ownership - do certain speakers focus on specific topics?
                    4. Public participation - how engaged are members of the public vs. officials?
                    
                    Format your response as JSON with this structure:
                    {{
                        "power_dynamics": {{
                            "dominant_speakers": ["Names"],
                            "observation": "Brief analysis"
                        }},
                        "interaction_patterns": {{
                            "pattern": "Description of how speakers interact",
                            "notable_exchanges": ["Brief descriptions"]
                        }},
                        "topic_ownership": [
                            {{
                                "speaker": "Name",
                                "topics": ["Topics this speaker focuses on"]
                            }}
                        ],
                        "public_participation": {{
                            "level": "High/Medium/Low",
                            "observation": "Analysis of public participation"
                        }}
                    }}""",
                    "agent": "speaker_analysis_agent",
                    "stream": False
                }
                
                speaker_response = await self.call_openai_agent(speaker_prompt)
                
                try:
                    speaker_analysis = json.loads(speaker_response.result)
                    result['speaker_analysis'].update(speaker_analysis)
                except:
                    result['speaker_analysis']['patterns'] = {"error": "Failed to parse response", "raw": speaker_response.result}
            
            except Exception as e:
                print(f"  Error in speaker analysis: {str(e)}")
                result['speaker_analysis']['patterns'] = {"error": str(e)}
        
        # 4. Emergent pattern detection - look for unexpected patterns
        try:
            pattern_prompt = {
                "input": f"""Analyze this Manhattan, Kansas city council meeting transcript for emergent patterns and insights.
                
                Based on the transcript:
                {all_text[:self.chunk_size]}
                
                Look for:
                1. Recurring themes or concerns that might not be obvious at first glance
                2. Underlying tensions or conflicts
                3. Decision-making patterns
                4. Linguistic patterns (formality, technical language, emotional tone)
                5. Any other patterns or insights that emerge from careful analysis
                
                Don't be constrained by predetermined categories - be creative in identifying patterns.
                
                Format your response as JSON:
                {{
                    "recurring_themes": [
                        {{
                            "theme": "Theme description",
                            "evidence": "How this manifests in the transcript"
                        }}
                    ],
                    "underlying_tensions": [
                        {{
                            "tension": "Description of tension",
                            "between": ["Parties involved"],
                            "context": "Brief explanation"
                        }}
                    ],
                    "decision_patterns": {{
                        "pattern": "How decisions seem to be made",
                        "observation": "Analysis of the decision-making process"
                    }},
                    "language_patterns": {{
                        "formality": "High/Medium/Low",
                        "technical_density": "High/Medium/Low",
                        "emotional_tone": "Description",
                        "notable_features": ["Any distinctive linguistic features"]
                    }},
                    "other_insights": [
                        "Any other patterns or insights noticed"
                    ]
                }}""",
                "agent": "pattern_detection_agent",
                "stream": False
            }
            
            pattern_response = await self.call_openai_agent(pattern_prompt)
            
            try:
                result['emergent_patterns'] = json.loads(pattern_response.result)
            except:
                result['emergent_patterns'] = {"error": "Failed to parse response", "raw": pattern_response.result}
        
        except Exception as e:
            print(f"  Error in pattern detection: {str(e)}")
            result['emergent_patterns'] = {"error": str(e)}
        
        return result

    def _combine_chunk_results(self, chunk_results):
        """Intelligently combine results from multiple transcript chunks."""
        # Create a structure for the combined results
        combined = {
            'open_ended_analysis': {
                'primary_themes': [],
                'significant_discussions': [],
                'unexpected_elements': []
            },
            'directed_analysis': {
                'land_use': {
                    'present': False,
                    'significance': 'Low',
                    'context': '',
                    'notable_points': []
                },
                'economic_development': {
                    'present': False,
                    'significance': 'Low',
                    'context': '',
                    'notable_points': []
                },
                'taxes_budget': {
                    'present': False,
                    'significance': 'Low',
                    'context': '',
                    'notable_points': []
                },
                'other_significant_topics': []
            },
            'speaker_analysis': {
                'speakers': {
                    'count': 0,
                    'speaking_time': {}
                },
                'power_dynamics': {
                    'dominant_speakers': [],
                    'observation': ''
                },
                'interaction_patterns': {
                    'pattern': '',
                    'notable_exchanges': []
                },
                'topic_ownership': [],
                'public_participation': {
                    'level': 'Unknown',
                    'observation': ''
                }
            },
            'emergent_patterns': {
                'recurring_themes': [],
                'underlying_tensions': [],
                'decision_patterns': {
                    'pattern': '',
                    'observation': ''
                },
                'language_patterns': {
                    'formality': 'Medium',
                    'technical_density': 'Medium',
                    'emotional_tone': '',
                    'notable_features': []
                },
                'other_insights': []
            }
        }
        
        # Track significance scores for focus areas
        significance_map = {'High': 3, 'Medium': 2, 'Low': 1, 'Unknown': 0}
        focus_areas = {
            'land_use': {'scores': [], 'contexts': [], 'points': set()},
            'economic_development': {'scores': [], 'contexts': [], 'points': set()},
            'taxes_budget': {'scores': [], 'contexts': [], 'points': set()}
        }
        
        # Collect theme mentions
        theme_mentions = Counter()
        
        # Process each chunk result
        for result in chunk_results:
            # Skip chunks with errors
            if 'error' in result and not any(k for k in result.keys() if k != 'error'):
                continue
            
            # Process open-ended analysis
            if 'open_ended_analysis' in result and isinstance(result['open_ended_analysis'], dict):
                # Collect primary themes
                if 'primary_themes' in result['open_ended_analysis'] and isinstance(result['open_ended_analysis']['primary_themes'], list):
                    for theme in result['open_ended_analysis']['primary_themes']:
                        if isinstance(theme, dict) and 'theme' in theme:
                            theme_mentions[theme['theme']] += 1
                            
                    combined['open_ended_analysis']['primary_themes'].extend(result['open_ended_analysis']['primary_themes'])
                
                # Collect significant discussions
                if 'significant_discussions' in result['open_ended_analysis'] and isinstance(result['open_ended_analysis']['significant_discussions'], list):
                    combined['open_ended_analysis']['significant_discussions'].extend(result['open_ended_analysis']['significant_discussions'])
                
                # Collect unexpected elements
                if 'unexpected_elements' in result['open_ended_analysis'] and isinstance(result['open_ended_analysis']['unexpected_elements'], list):
                    combined['open_ended_analysis']['unexpected_elements'].extend(result['open_ended_analysis']['unexpected_elements'])
            
            # Process directed analysis
            if 'directed_analysis' in result and isinstance(result['directed_analysis'], dict):
                # Process focus areas
                for area in focus_areas:
                    if area in result['directed_analysis'] and isinstance(result['directed_analysis'][area], dict):
                        area_data = result['directed_analysis'][area]
                        
                        # Update presence
                        if area_data.get('present', False):
                            combined['directed_analysis'][area]['present'] = True
                        
                        # Collect significance scores
                        if 'significance' in area_data:
                            focus_areas[area]['scores'].append(significance_map.get(area_data['significance'], 0))
                        
                        # Collect contexts
                        if 'context' in area_data and area_data['context']:
                            focus_areas[area]['contexts'].append(area_data['context'])
                        
                        # Collect notable points
                        if 'notable_points' in area_data and isinstance(area_data['notable_points'], list):
                            focus_areas[area]['points'].update(area_data['notable_points'])
                
                # Collect other significant topics
                if 'other_significant_topics' in result['directed_analysis'] and isinstance(result['directed_analysis']['other_significant_topics'], list):
                    combined['directed_analysis']['other_significant_topics'].extend(result['directed_analysis']['other_significant_topics'])
            
            # Process speaker analysis
            if 'speaker_analysis' in result and isinstance(result['speaker_analysis'], dict):
                # Merge speaker count and time data
                if 'speakers' in result['speaker_analysis'] and isinstance(result['speaker_analysis']['speakers'], dict):
                    combined['speaker_analysis']['speakers']['count'] = max(
                        combined['speaker_analysis']['speakers']['count'],
                        result['speaker_analysis']['speakers'].get('count', 0)
                    )
                    
                    # Merge speaking time
                    for speaker, time in result['speaker_analysis']['speakers'].get('speaking_time', {}).items():
                        if speaker in combined['speaker_analysis']['speakers']['speaking_time']:
                            combined['speaker_analysis']['speakers']['speaking_time'][speaker] += time
                        else:
                            combined['speaker_analysis']['speakers']['speaking_time'][speaker] = time
                
                # Collect power dynamics
                if 'power_dynamics' in result['speaker_analysis'] and isinstance(result['speaker_analysis']['power_dynamics'], dict):
                    # Extend dominant speakers list
                    combined['speaker_analysis']['power_dynamics']['dominant_speakers'].extend(
                        [s for s in result['speaker_analysis']['power_dynamics'].get('dominant_speakers', [])
                         if s not in combined['speaker_analysis']['power_dynamics']['dominant_speakers']]
                    )
                    
                    # Combine observations
                    if result['speaker_analysis']['power_dynamics'].get('observation'):
                        if combined['speaker_analysis']['power_dynamics']['observation']:
                            combined['speaker_analysis']['power_dynamics']['observation'] += f"; {result['speaker_analysis']['power_dynamics']['observation']}"
                        else:
                            combined['speaker_analysis']['power_dynamics']['observation'] = result['speaker_analysis']['power_dynamics']['observation']
                
                # Collect interaction patterns
                if 'interaction_patterns' in result['speaker_analysis'] and isinstance(result['speaker_analysis']['interaction_patterns'], dict):
                    # Combine pattern descriptions
                    if result['speaker_analysis']['interaction_patterns'].get('pattern'):
                        if combined['speaker_analysis']['interaction_patterns']['pattern']:
                            combined['speaker_analysis']['interaction_patterns']['pattern'] += f"; {result['speaker_analysis']['interaction_patterns']['pattern']}"
                        else:
                            combined['speaker_analysis']['interaction_patterns']['pattern'] = result['speaker_analysis']['interaction_patterns']['pattern']
                    
                    # Extend notable exchanges
                    combined['speaker_analysis']['interaction_patterns']['notable_exchanges'].extend(
                        result['speaker_analysis']['interaction_patterns'].get('notable_exchanges', [])
                    )
                
                # Collect topic ownership
                if 'topic_ownership' in result['speaker_analysis'] and isinstance(result['speaker_analysis']['topic_ownership'], list):
                    # Create a map of existing speaker -> topics
                    existing_speakers = {item['speaker']: item['topics'] for item in combined['speaker_analysis']['topic_ownership'] if 'speaker' in item and 'topics' in item}
                    
                    # Process new topic ownership data
                    for item in result['speaker_analysis']['topic_ownership']:
                        if 'speaker' in item and 'topics' in item:
                            if item['speaker'] in existing_speakers:
                                # Add new topics
                                existing_speakers[item['speaker']].extend([t for t in item['topics'] if t not in existing_speakers[item['speaker']]])
                            else:
                                # Add new speaker
                                combined['speaker_analysis']['topic_ownership'].append(item)
                
                # Update public participation
                if 'public_participation' in result['speaker_analysis'] and isinstance(result['speaker_analysis']['public_participation'], dict):
                    # Prioritize higher participation levels
                    current_level = significance_map.get(combined['speaker_analysis']['public_participation'].get('level', 'Unknown'), 0)
                    new_level = significance_map.get(result['speaker_analysis']['public_participation'].get('level', 'Unknown'), 0)
                    
                    if new_level > current_level:
                        combined['speaker_analysis']['public_participation']['level'] = result['speaker_analysis']['public_participation'].get('level', 'Unknown')
                    
                    # Combine observations
                    if result['speaker_analysis']['public_participation'].get('observation'):
                        if combined['speaker_analysis']['public_participation']['observation']:
                            combined['speaker_analysis']['public_participation']['observation'] += f"; {result['speaker_analysis']['public_participation']['observation']}"
                        else:
                            combined['speaker_analysis']['public_participation']['observation'] = result['speaker_analysis']['public_participation']['observation']
            
            # Process emergent patterns
            if 'emergent_patterns' in result and isinstance(result['emergent_patterns'], dict):
                # Collect recurring themes
                if 'recurring_themes' in result['emergent_patterns'] and isinstance(result['emergent_patterns']['recurring_themes'], list):
                    combined['emergent_patterns']['recurring_themes'].extend(result['emergent_patterns']['recurring_themes'])
                
                # Collect underlying tensions
                if 'underlying_tensions' in result['emergent_patterns'] and isinstance(result['emergent_patterns']['underlying_tensions'], list):
                    combined['emergent_patterns']['underlying_tensions'].extend(result['emergent_patterns']['underlying_tensions'])
                
                # Update decision patterns
                if 'decision_patterns' in result['emergent_patterns'] and isinstance(result['emergent_patterns']['decision_patterns'], dict):
                    if result['emergent_patterns']['decision_patterns'].get('pattern'):
                        if combined['emergent_patterns']['decision_patterns']['pattern']:
                            combined['emergent_patterns']['decision_patterns']['pattern'] += f"; {result['emergent_patterns']['decision_patterns']['pattern']}"
                        else:
                            combined['emergent_patterns']['decision_patterns']['pattern'] = result['emergent_patterns']['decision_patterns']['pattern']
                    
                    if result['emergent_patterns']['decision_patterns'].get('observation'):
                        if combined['emergent_patterns']['decision_patterns']['observation']:
                            combined['emergent_patterns']['decision_patterns']['observation'] += f"; {result['emergent_patterns']['decision_patterns']['observation']}"
                        else:
                            combined['emergent_patterns']['decision_patterns']['observation'] = result['emergent_patterns']['decision_patterns']['observation']
                
                # Update language patterns
                if 'language_patterns' in result['emergent_patterns'] and isinstance(result['emergent_patterns']['language_patterns'], dict):
                    # Update formality (prioritize more extreme values)
                    formality_map = {'High': 3, 'Medium': 2, 'Low': 1}
                    current_formality = formality_map.get(combined['emergent_patterns']['language_patterns']['formality'], 2)
                    new_formality = formality_map.get(result['emergent_patterns']['language_patterns'].get('formality', 'Medium'), 2)
                    
                    if abs(new_formality - 2) > abs(current_formality - 2):
                        combined['emergent_patterns']['language_patterns']['formality'] = result['emergent_patterns']['language_patterns'].get('formality', 'Medium')
                    
                    # Update technical density (similar approach)
                    tech_map = {'High': 3, 'Medium': 2, 'Low': 1}
                    current_tech = tech_map.get(combined['emergent_patterns']['language_patterns']['technical_density'], 2)
                    new_tech = tech_map.get(result['emergent_patterns']['language_patterns'].get('technical_density', 'Medium'), 2)
                    
                    if abs(new_tech - 2) > abs(current_tech - 2):
                        combined['emergent_patterns']['language_patterns']['technical_density'] = result['emergent_patterns']['language_patterns'].get('technical_density', 'Medium')
                    
                    # Combine emotional tone
                    if result['emergent_patterns']['language_patterns'].get('emotional_tone'):
                        if combined['emergent_patterns']['language_patterns']['emotional_tone']:
                            combined['emergent_patterns']['language_patterns']['emotional_tone'] += f"; {result['emergent_patterns']['language_patterns']['emotional_tone']}"
                        else:
                            combined['emergent_patterns']['language_patterns']['emotional_tone'] = result['emergent_patterns']['language_patterns']['emotional_tone']
                    
                    # Extend notable features
                    combined['emergent_patterns']['language_patterns']['notable_features'].extend(
                        [f for f in result['emergent_patterns']['language_patterns'].get('notable_features', [])
                         if f not in combined['emergent_patterns']['language_patterns']['notable_features']]
                    )
                
                # Collect other insights
                if 'other_insights' in result['emergent_patterns'] and isinstance(result['emergent_patterns']['other_insights'], list):
                    combined['emergent_patterns']['other_insights'].extend(result['emergent_patterns']['other_insights'])
        
        # Process collected data
        
        # Sort primary themes by mention frequency and limit to top 10
        sorted_themes = []
        seen_themes = set()
        
        # First, add most mentioned themes from counter
        for theme, count in theme_mentions.most_common():
            if theme not in seen_themes:
                # Find the full theme entry
                for result in chunk_results:
                    if 'open_ended_analysis' in result and 'primary_themes' in result['open_ended_analysis']:
                        for theme_entry in result['open_ended_analysis']['primary_themes']:
                            if isinstance(theme_entry, dict) and theme_entry.get('theme') == theme:
                                sorted_themes.append(theme_entry)
                                seen_themes.add(theme)
                                break
            
            if len(sorted_themes) >= 10:
                break
        
        # Add any remaining themes up to 10
        for result in chunk_results:
            if 'open_ended_analysis' in result and 'primary_themes' in result['open_ended_analysis']:
                for theme_entry in result['open_ended_analysis']['primary_themes']:
                    if isinstance(theme_entry, dict) and 'theme' in theme_entry:
                        if theme_entry['theme'] not in seen_themes:
                            sorted_themes.append(theme_entry)
                            seen_themes.add(theme_entry['theme'])
                            
                            if len(sorted_themes) >= 10:
                                break
                
                if len(sorted_themes) >= 10:
                    break
        
        combined['open_ended_analysis']['primary_themes'] = sorted_themes
        
        # Update focus areas with collected data
        for area in focus_areas:
            # Calculate significance
            if focus_areas[area]['scores']:
                avg_score = sum(focus_areas[area]['scores']) / len(focus_areas[area]['scores'])
                sig_reverse_map = {3: 'High', 2: 'Medium', 1: 'Low', 0: 'Unknown'}
                combined['directed_analysis'][area]['significance'] = sig_reverse_map.get(round(avg_score), 'Low')
            
            # Combine contexts (limit to prevent excessive text)
            contexts = focus_areas[area]['contexts']
            if contexts:
                combined['directed_analysis'][area]['context'] = "; ".join(contexts[:3])
            
            # Add notable points
            combined['directed_analysis'][area]['notable_points'] = list(focus_areas[area]['points'])
        
        # Deduplicate other significant topics
        seen_topics = set()
        unique_topics = []
        
        for topic in combined['directed_analysis']['other_significant_topics']:
            if isinstance(topic, dict) and 'topic' in topic:
                if topic['topic'] not in seen_topics:
                    unique_topics.append(topic)
                    seen_topics.add(topic['topic'])
        
        combined['directed_analysis']['other_significant_topics'] = unique_topics
        
        # Deduplicate recurring themes
        seen_themes = set()
        unique_themes = []
        
        for theme in combined['emergent_patterns']['recurring_themes']:
            if isinstance(theme, dict) and 'theme' in theme:
                if theme['theme'] not in seen_themes:
                    unique_themes.append(theme)
                    seen_themes.add(theme['theme'])
        
        combined['emergent_patterns']['recurring_themes'] = unique_themes
        
        # Deduplicate underlying tensions
        seen_tensions = set()
        unique_tensions = []
        
        for tension in combined['emergent_patterns']['underlying_tensions']:
            if isinstance(tension, dict) and 'tension' in tension:
                if tension['tension'] not in seen_tensions:
                    unique_tensions.append(tension)
                    seen_tensions.add(tension['tension'])
        
        combined['emergent_patterns']['underlying_tensions'] = unique_tensions
        
        return combined

    async def analyze_all_transcripts(self):
        """Analyze all transcripts and save results."""
        print("Analyzing all transcripts...")
        
        # Create directory for full results
        os.makedirs('results', exist_ok=True)
        
        # Create a list of files sorted by size (smallest first)
        files_by_size = sorted(
            [(filename, transcript_data) for filename, transcript_data in self.transcripts.items()],
            key=lambda x: x[1]['file_size']
        )
        
        # Process all transcripts
        for filename, transcript_data in tqdm(files_by_size):
            try:
                # Check if this file has already been analyzed
                safe_filename = filename.replace("/", "_").replace("\\", "_")
                result_file = f'results/{safe_filename}_analysis.json'
                if os.path.exists(result_file):
                    print(f"Loading existing analysis for {filename}...")
                    try:
                        with open(result_file, 'r') as f:
                            analysis = json.load(f)
                    except json.JSONDecodeError:
                        print(f"  Error loading existing analysis file for {filename}, will reanalyze")
                        analysis = await self.analyze_transcript(filename, transcript_data)
                        
                        # Save individual result
                        with open(result_file, 'w') as f:
                            json.dump(analysis, f, indent=2)
                else:
                    print(f"\nAnalyzing {filename} ({transcript_data['file_size']} chars)...")
                    analysis = await self.analyze_transcript(filename, transcript_data)
                    
                    # Save individual result
                    with open(result_file, 'w') as f:
                        json.dump(analysis, f, indent=2)
                
                # Store in meeting analyses
                self.meeting_analyses[filename] = analysis
                
                # Save progress
                with open('analysis_progress.json', 'w') as f:
                    json.dump({
                        'completed_files': list(self.meeting_analyses.keys()),
                        'remaining_files': [f for f, _ in files_by_size if f not in self.meeting_analyses],
                        'timestamp': datetime.now().isoformat()
                    }, f, indent=2)
                
                # Add a delay between files to avoid API rate limits
                await asyncio.sleep(2)
                
            except Exception as e:
                print(f"Error analyzing {filename}: {str(e)}")
                self.meeting_analyses[filename] = {'error': str(e)}
                
                # Save error information
                with open(f'results/{safe_filename}_error.json', 'w') as f:
                    json.dump({'error': str(e)}, f, indent=2)
        
        print(f"Analyzed {len(self.meeting_analyses)} transcript files.")
        return self.meeting_analyses

    async def detect_global_patterns(self):
        """Detect patterns across all meetings using OpenAI agent."""
        print("Detecting global patterns across all meetings...")
        
        # Prepare data for global analysis
        global_data = {
            'meeting_count': len(self.meeting_analyses),
            'date_range': {
                'start': min([data['date'] for _, data in self.transcripts.items() if data['date'] != 'Unknown']),
                'end': max([data['date'] for _, data in self.transcripts.items() if data['date'] != 'Unknown'])
            },
            'topics': {},
            'speakers': {},
            'patterns': {}
        }
        
        # Collect primary themes across all meetings
        all_themes = []
        for filename, analysis in self.meeting_analyses.items():
            if 'open_ended_analysis' in analysis and 'primary_themes' in analysis['open_ended_analysis']:
                for theme in analysis['open_ended_analysis']['primary_themes']:
                    if isinstance(theme, dict) and 'theme' in theme:
                        all_themes.append({
                            'theme': theme['theme'],
                            'description': theme.get('description', ''),
                            'meeting': filename,
                            'date': self.transcripts[filename]['date']
                        })
        
        # Collect focus areas data
        focus_areas_data = {
            'land_use': [],
            'economic_development': [],
            'taxes_budget': []
        }
        
        for filename, analysis in self.meeting_analyses.items():
            if 'directed_analysis' in analysis:
                for area in focus_areas_data:
                    if area in analysis['directed_analysis'] and analysis['directed_analysis'][area].get('present', False):
                        focus_areas_data[area].append({
                            'meeting': filename,
                            'date': self.transcripts[filename]['date'],
                            'significance': analysis['directed_analysis'][area].get('significance', 'Low'),
                            'context': analysis['directed_analysis'][area].get('context', '')
                        })
        
        # Collect speaker data
        all_speakers = set()
        speaker_mentions = Counter()
        
        for filename, analysis in self.meeting_analyses.items():
            if 'speaker_analysis' in analysis and 'speakers' in analysis['speaker_analysis']:
                for speaker in analysis['speaker_analysis']['speakers'].get('speaking_time', {}).keys():
                    all_speakers.add(speaker)
                    speaker_mentions[speaker] += 1
        
        # Collect recurring themes and tensions
        recurring_themes = []
        underlying_tensions = []
        
        for filename, analysis in self.meeting_analyses.items():
            if 'emergent_patterns' in analysis:
                if 'recurring_themes' in analysis['emergent_patterns']:
                    for theme in analysis['emergent_patterns']['recurring_themes']:
                        if isinstance(theme, dict) and 'theme' in theme:
                            recurring_themes.append({
                                'theme': theme['theme'],
                                'evidence': theme.get('evidence', ''),
                                'meeting': filename,
                                'date': self.transcripts[filename]['date']
                            })
                
                if 'underlying_tensions' in analysis['emergent_patterns']:
                    for tension in analysis['emergent_patterns']['underlying_tensions']:
                        if isinstance(tension, dict) and 'tension' in tension:
                            underlying_tensions.append({
                                'tension': tension['tension'],
                                'between': tension.get('between', []),
                                'context': tension.get('context', ''),
                                'meeting': filename,
                                'date': self.transcripts[filename]['date']
                            })
        
        # Prepare data for global pattern analysis
        global_data['topics'] = {
            'all_themes': all_themes[:100],  # Limit to keep request size manageable
            'focus_areas': focus_areas_data
        }
        
        global_data['speakers'] = {
            'unique_count': len(all_speakers),
            'frequent_speakers': [{'speaker': s, 'mentions': c} for s, c in speaker_mentions.most_common(20)]
        }
        
        global_data['patterns'] = {
            'recurring_themes': recurring_themes[:50],  # Limit to keep request size manageable
            'underlying_tensions': underlying_tensions[:50]  # Limit to keep request size manageable
        }
        
        # Add temporal data
        years = sorted(set(data['year'] for _, data in self.transcripts.items() if data['year'] != 'Unknown'))
        year_data = []
        
        for year in years:
            year_meetings = [filename for filename, data in self.transcripts.items() if data['year'] == year]
            
            land_use_count = sum(1 for filename in year_meetings if 
                             filename in self.meeting_analyses and
                             'directed_analysis' in self.meeting_analyses[filename] and
                             'land_use' in self.meeting_analyses[filename]['directed_analysis'] and
                             self.meeting_analyses[filename]['directed_analysis']['land_use'].get('present', False))
            
            econ_dev_count = sum(1 for filename in year_meetings if 
                              filename in self.meeting_analyses and
                              'directed_analysis' in self.meeting_analyses[filename] and
                              'economic_development' in self.meeting_analyses[filename]['directed_analysis'] and
                              self.meeting_analyses[filename]['directed_analysis']['economic_development'].get('present', False))
            
            tax_budget_count = sum(1 for filename in year_meetings if 
                                filename in self.meeting_analyses and
                                'directed_analysis' in self.meeting_analyses[filename] and
                                'taxes_budget' in self.meeting_analyses[filename]['directed_analysis'] and
                                self.meeting_analyses[filename]['directed_analysis']['taxes_budget'].get('present', False))
            
            year_data.append({
                'year': year,
                'meeting_count': len(year_meetings),
                'land_use_count': land_use_count,
                'economic_development_count': econ_dev_count,
                'taxes_budget_count': tax_budget_count
            })
        
        global_data['temporal'] = {
            'years': year_data
        }
        
        try:
            # Call OpenAI agent to analyze global patterns
            global_prompt = {
                "input": f"""Analyze the patterns across {len(self.meeting_analyses)} Manhattan, Kansas city council meetings from {global_data['date_range']['start']} to {global_data['date_range']['end']}.

                Here is aggregated data from these meetings:
                {json.dumps(global_data, indent=2)}
                
                Identify:
                1. Major longitudinal trends - how have discussions evolved over time?
                2. Recurring topics across meetings - what issues keep coming up?
                3. Unexpected patterns - what surprising insights emerge from this data?
                4. Topic relationships - which topics tend to be discussed together?
                5. Speaker dynamics - are there patterns in who speaks about what?
                6. Decision-making patterns - how does this council tend to make decisions?
                7. Any other significant patterns across these meetings
                
                Provide your analysis in a comprehensive JSON format with detailed explanations for each pattern identified.
                Be creative and perceptive in identifying patterns that might not be immediately obvious.
                """,
                "agent": "global_pattern_detection_agent",
                "stream": False
            }
            
            print("Calling OpenAI agent for global pattern analysis...")
            global_response = await self.call_openai_agent(global_prompt)
            
            try:
                self.global_patterns = json.loads(global_response.result)
            except:
                self.global_patterns = {"error": "Failed to parse response", "raw": global_response.result}
            
            # Save global patterns
            with open('results/global_patterns.json', 'w') as f:
                json.dump(self.global_patterns, f, indent=2)
            
            print("Global pattern analysis complete.")
            return self.global_patterns
            
        except Exception as e:
            print(f"Error in global pattern analysis: {str(e)}")
            self.global_patterns = {"error": str(e)}
            return self.global_patterns

    def build_topic_network(self):
        """Build a network of relationships between topics discussed in meetings."""
        print("Building topic network...")
        
        # Create a graph
        G = nx.Graph()
        
        # Track topic co-occurrences
        topic_pairs = Counter()
        
        # Collect all topics
        all_topics = set()
        
        for filename, analysis in self.meeting_analyses.items():
            # Get topics from primary themes
            meeting_topics = set()
            
            if 'open_ended_analysis' in analysis and 'primary_themes' in analysis['open_ended_analysis']:
                for theme in analysis['open_ended_analysis']['primary_themes']:
                    if isinstance(theme, dict) and 'theme' in theme:
                        meeting_topics.add(theme['theme'])
                        all_topics.add(theme['theme'])
            
            # Get topics from directed analysis
            if 'directed_analysis' in analysis:
                # Add focus areas
                for area in ['land_use', 'economic_development', 'taxes_budget']:
                    if area in analysis['directed_analysis'] and analysis['directed_analysis'][area].get('present', False):
                        meeting_topics.add(area.replace('_', ' '))
                        all_topics.add(area.replace('_', ' '))
                
                # Add other significant topics
                if 'other_significant_topics' in analysis['directed_analysis']:
                    for topic in analysis['directed_analysis']['other_significant_topics']:
                        if isinstance(topic, dict) and 'topic' in topic:
                            meeting_topics.add(topic['topic'])
                            all_topics.add(topic['topic'])
            
            # Count co-occurrences of topics within the same meeting
            topics_list = list(meeting_topics)
            for i in range(len(topics_list)):
                for j in range(i+1, len(topics_list)):
                    topic_pair = tuple(sorted([topics_list[i], topics_list[j]]))
                    topic_pairs[topic_pair] += 1
        
        # Add nodes to the graph
        for topic in all_topics:
            G.add_node(topic)
        
        # Add edges based on co-occurrence
        for (topic1, topic2), weight in topic_pairs.items():
            if weight >= 2:  # Only add edge if topics co-occur in at least 2 meetings
                G.add_edge(topic1, topic2, weight=weight)
        
        # Store the network
        self.topic_network = G
        
        # Visualize the network
        plt.figure(figsize=(15, 15))
        
        # Calculate node sizes based on frequency
        topic_freq = Counter()
        for filename, analysis in self.meeting_analyses.items():
            if 'open_ended_analysis' in analysis and 'primary_themes' in analysis['open_ended_analysis']:
                for theme in analysis['open_ended_analysis']['primary_themes']:
                    if isinstance(theme, dict) and 'theme' in theme:
                        topic_freq[theme['theme']] += 1
        
        node_sizes = [100 + 20 * topic_freq.get(topic, 0) for topic in G.nodes()]
        
        # Calculate edge widths
        edge_widths = [0.5 + 0.5 * G[u][v]['weight'] for u, v in G.edges()]
        
        # Use a force-directed layout
        pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42)
        
        # Draw the network
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, alpha=0.7, node_color='skyblue')
        nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.5, edge_color='gray')
        nx.draw_networkx_labels(G, pos, font_size=8, font_family='sans-serif')
        
        plt.title('Topic Co-occurrence Network')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('visualizations/topic_network.png', dpi=300)
        plt.close()
        
        print("Topic network analysis complete.")
        return G

    def build_speaker_network(self):
        """Build a network of relationships between speakers in meetings."""
        print("Building speaker network...")
        
        # Create a graph
        G = nx.Graph()
        
        # Track speaker co-occurrences
        speaker_pairs = Counter()
        
        # Collect all speakers
        all_speakers = set()
        
        for filename, analysis in self.meeting_analyses.items():
            # Get speakers from speaker analysis
            meeting_speakers = set()
            
            if 'speaker_analysis' in analysis and 'speakers' in analysis['speaker_analysis']:
                for speaker in analysis['speaker_analysis']['speakers'].get('speaking_time', {}).keys():
                    meeting_speakers.add(speaker)
                    all_speakers.add(speaker)
            
            # Count co-occurrences of speakers within the same meeting
            speakers_list = list(meeting_speakers)
            for i in range(len(speakers_list)):
                for j in range(i+1, len(speakers_list)):
                    speaker_pair = tuple(sorted([speakers_list[i], speakers_list[j]]))
                    speaker_pairs[speaker_pair] += 1
        
        # Add nodes to the graph
        for speaker in all_speakers:
            G.add_node(speaker)
        
        # Add edges based on co-occurrence
        for (speaker1, speaker2), weight in speaker_pairs.items():
            if weight >= 3:  # Only add edge if speakers co-occur in at least 3 meetings
                G.add_edge(speaker1, speaker2, weight=weight)
        
        # Store the network
        self.speaker_network = G
        
        # Visualize the network
        plt.figure(figsize=(15, 15))
        
        # Calculate node sizes based on frequency
        speaker_freq = Counter()
        for filename, analysis in self.meeting_analyses.items():
            if 'speaker_analysis' in analysis and 'speakers' in analysis['speaker_analysis']:
                for speaker in analysis['speaker_analysis']['speakers'].get('speaking_time', {}).keys():
                    speaker_freq[speaker] += 1
        
        node_sizes = [50 + 10 * speaker_freq.get(speaker, 0) for speaker in G.nodes()]
        
        # Calculate edge widths
        edge_widths = [0.5 + 0.5 * G[u][v]['weight'] for u, v in G.edges()]
        
        # Use a force-directed layout
        pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42)
        
        # Draw the network
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, alpha=0.7, node_color='lightgreen')
        nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.5, edge_color='gray')
        nx.draw_networkx_labels(G, pos, font_size=8, font_family='sans-serif')
        
        plt.title('Speaker Co-occurrence Network')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('visualizations/speaker_network.png', dpi=300)
        plt.close()
        
        print("Speaker network analysis complete.")
        return G

    def analyze_temporal_trends(self):
        """Analyze how topics and patterns evolve over time."""
        print("Analyzing temporal trends...")
        
        # Get years in chronological order
        years = sorted(set(data['year'] for _, data in self.transcripts.items() if data['year'] != 'Unknown'))
        
        # Track topic frequency by year
        topic_by_year = {year: Counter() for year in years}
        
        # Track focus areas by year
        focus_areas_by_year = {
            'land_use': {year: {'count': 0, 'high': 0, 'medium': 0, 'low': 0} for year in years},
            'economic_development': {year: {'count': 0, 'high': 0, 'medium': 0, 'low': 0} for year in years},
            'taxes_budget': {year: {'count': 0, 'high': 0, 'medium': 0, 'low': 0} for year in years}
        }
        
        # Track speaking patterns by year
        speaking_equality_by_year = {year: [] for year in years}
        public_participation_by_year = {year: Counter() for year in years}
        
        # Analyze each meeting
        for filename, analysis in self.meeting_analyses.items():
            year = self.transcripts[filename]['year']
            if year == 'Unknown':
                continue
            
            # Extract topics
            if 'open_ended_analysis' in analysis and 'primary_themes' in analysis['open_ended_analysis']:
                for theme in analysis['open_ended_analysis']['primary_themes']:
                    if isinstance(theme, dict) and 'theme' in theme:
                        topic_by_year[year][theme['theme']] += 1
            
            # Extract focus areas
            if 'directed_analysis' in analysis:
                for area in focus_areas_by_year:
                    if area in analysis['directed_analysis'] and analysis['directed_analysis'][area].get('present', False):
                        focus_areas_by_year[area][year]['count'] += 1
                        
                        significance = analysis['directed_analysis'][area].get('significance', 'Low').lower()
                        if significance == 'high':
                            focus_areas_by_year[area][year]['high'] += 1
                        elif significance == 'medium':
                            focus_areas_by_year[area][year]['medium'] += 1
                        else:
                            focus_areas_by_year[area][year]['low'] += 1
            
            # Extract speaker equality
            if 'speaker_analysis' in analysis and 'speakers' in analysis['speaker_analysis'] and 'speaking_time' in analysis['speaker_analysis']['speakers']:
                speaking_times = list(analysis['speaker_analysis']['speakers']['speaking_time'].values())
                if speaking_times:
                    # Calculate Gini coefficient
                    gini = self._calculate_gini(speaking_times)
                    speaking_equality_by_year[year].append(gini)
            
            # Extract public participation
            if 'speaker_analysis' in analysis and 'public_participation' in analysis['speaker_analysis']:
                level = analysis['speaker_analysis']['public_participation'].get('level', 'Unknown')
                public_participation_by_year[year][level] += 1
        
        # Store temporal trends
        self.temporal_trends = {
            'years': years,
            'topic_by_year': topic_by_year,
            'focus_areas_by_year': focus_areas_by_year,
            'speaking_equality_by_year': speaking_equality_by_year,
            'public_participation_by_year': public_participation_by_year
        }
        
        # Visualize temporal trends
        
        # 1. Focus Areas Over Time
        plt.figure(figsize=(15, 10))
        
        for i, (area, data) in enumerate(focus_areas_by_year.items()):
            plt.subplot(3, 1, i+1)
            
            total_counts = [data[year]['count'] for year in years]
            high_counts = [data[year]['high'] for year in years]
            
            plt.bar(years, total_counts, alpha=0.7, label='Total Mentions')
            plt.bar(years, high_counts, alpha=0.7, label='High Significance')
            
            plt.title(f'{area.replace("_", " ").title()} Discussions Over Time')
            plt.ylabel('Number of Meetings')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig('visualizations/focus_areas_over_time.png')
        plt.close()
        
        # 2. Speaking Equality Over Time
        plt.figure(figsize=(12, 8))
        
        avg_gini_by_year = [np.mean(speaking_equality_by_year[year]) if speaking_equality_by_year[year] else 0 for year in years]
        
        plt.bar(years, avg_gini_by_year)
        plt.axhline(y=np.mean(avg_gini_by_year), color='r', linestyle='--', label='Overall Average')
        
        plt.title('Speaker Equality Over Time (Gini Coefficient)')
        plt.ylabel('Average Gini Coefficient\n(Higher = More Unequal)')
        plt.legend()
        plt.tight_layout()
        plt.savefig('visualizations/speaking_equality_over_time.png')
        plt.close()
        
        # 3. Top Topics WordCloud Over Time
        os.makedirs('visualizations/wordclouds', exist_ok=True)
        
        for year in years:
            if not topic_by_year[year]:
                continue
                
            wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=100).generate_from_frequencies(topic_by_year[year])
            
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title(f'Top Topics in {year}')
            plt.tight_layout()
            plt.savefig(f'visualizations/wordclouds/topics_{year}.png')
            plt.close()
        
        print("Temporal trends analysis complete.")
        return self.temporal_trends

    def detect_anomalies(self):
        """Detect anomalous meetings that stand out from the norm."""
        print("Detecting anomalous meetings...")
        
        # Collect metrics for each meeting
        meeting_metrics = {}
        
        for filename, analysis in self.meeting_analyses.items():
            metrics = {}
            
            # Number of speakers
            if 'speaker_analysis' in analysis and 'speakers' in analysis['speaker_analysis']:
                metrics['speaker_count'] = analysis['speaker_analysis']['speakers'].get('count', 0)
            else:
                metrics['speaker_count'] = 0
            
            # Speaking equality (Gini coefficient)
            if 'speaker_analysis' in analysis and 'speakers' in analysis['speaker_analysis'] and 'speaking_time' in analysis['speaker_analysis']['speakers']:
                speaking_times = list(analysis['speaker_analysis']['speakers']['speaking_time'].values())
                if speaking_times:
                    metrics['gini'] = self._calculate_gini(speaking_times)
                else:
                    metrics['gini'] = 0
            else:
                metrics['gini'] = 0
            
            # Number of topics
            if 'open_ended_analysis' in analysis and 'primary_themes' in analysis['open_ended_analysis']:
                metrics['topic_count'] = len(analysis['open_ended_analysis']['primary_themes'])
            else:
                metrics['topic_count'] = 0
            
            # Number of tensions
            if 'emergent_patterns' in analysis and 'underlying_tensions' in analysis['emergent_patterns']:
                metrics['tension_count'] = len(analysis['emergent_patterns']['underlying_tensions'])
            else:
                metrics['tension_count'] = 0
            
            # Focus areas presence
            for area in ['land_use', 'economic_development', 'taxes_budget']:
                if 'directed_analysis' in analysis and area in analysis['directed_analysis']:
                    metrics[f'{area}_present'] = 1 if analysis['directed_analysis'][area].get('present', False) else 0
                else:
                    metrics[f'{area}_present'] = 0
            
            meeting_metrics[filename] = metrics
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame.from_dict(meeting_metrics, orient='index')
        
        # Calculate z-scores for numerical columns
        numerical_cols = ['speaker_count', 'gini', 'topic_count', 'tension_count']
        for col in numerical_cols:
            if col in df.columns:
                df[f'{col}_zscore'] = (df[col] - df[col].mean()) / df[col].std()
        
        # Identify anomalous meetings (those with extreme z-scores)
        anomalous_meetings = []
        
        for filename, row in df.iterrows():
            anomaly_scores = []
            
            for col in [f'{c}_zscore' for c in numerical_cols if f'{c}_zscore' in df.columns]:
                if abs(row[col]) > 2:  # More than 2 standard deviations from the mean
                    anomaly_scores.append((col, row[col]))
            
            if anomaly_scores:
                date = self.transcripts[filename]['date']
                
                anomalous_meetings.append({
                    'filename': filename,
                    'date': date,
                    'anomaly_scores': anomaly_scores,
                    'metrics': {col: row[col] for col in numerical_cols if col in df.columns}
                })
        
        # Sort by the maximum absolute z-score
        anomalous_meetings.sort(key=lambda x: max(abs(score) for _, score in x['anomaly_scores']), reverse=True)
        
        # Store anomalies
        self.anomalies = anomalous_meetings[:10]  # Keep top 10
        
        # Visualize anomalies
        if anomalous_meetings:
            plt.figure(figsize=(15, 10))
            
            # Plot the distribution of topic counts with anomalies highlighted
            plt.subplot(2, 2, 1)
            sns.histplot(df['topic_count'], kde=True)
            for meeting in anomalous_meetings[:5]:
                plt.axvline(x=meeting['metrics']['topic_count'], color='r', linestyle='--', 
                            label=meeting['date'] if 'topic_count_zscore' in [s[0] for s in meeting['anomaly_scores']] else None)
            plt.title('Topic Count Distribution')
            plt.legend()
            
            # Plot the distribution of speaker counts with anomalies highlighted
            plt.subplot(2, 2, 2)
            sns.histplot(df['speaker_count'], kde=True)
            for meeting in anomalous_meetings[:5]:
                plt.axvline(x=meeting['metrics']['speaker_count'], color='r', linestyle='--',
                           label=meeting['date'] if 'speaker_count_zscore' in [s[0] for s in meeting['anomaly_scores']] else None)
            plt.title('Speaker Count Distribution')
            plt.legend()
            
            # Plot the distribution of Gini coefficients with anomalies highlighted
            plt.subplot(2, 2, 3)
            sns.histplot(df['gini'], kde=True)
            for meeting in anomalous_meetings[:5]:
                plt.axvline(x=meeting['metrics']['gini'], color='r', linestyle='--',
                           label=meeting['date'] if 'gini_zscore' in [s[0] for s in meeting['anomaly_scores']] else None)
            plt.title('Speaking Equality (Gini) Distribution')
            plt.legend()
            
            # Plot the distribution of tension counts with anomalies highlighted
            plt.subplot(2, 2, 4)
            sns.histplot(df['tension_count'], kde=True)
            for meeting in anomalous_meetings[:5]:
                plt.axvline(x=meeting['metrics']['tension_count'], color='r', linestyle='--',
                           label=meeting['date'] if 'tension_count_zscore' in [s[0] for s in meeting['anomaly_scores']] else None)
            plt.title('Tension Count Distribution')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig('visualizations/anomalous_meetings.png')
            plt.close()
        
        print(f"Detected {len(self.anomalies)} anomalous meetings.")
        return self.anomalies

    def _calculate_gini(self, values):
        """Calculate Gini coefficient as a measure of inequality."""
        if not values or sum(values) == 0:
            return 0
        
        # Sort values in ascending order
        sorted_values = sorted(values)
        n = len(sorted_values)
        
        # Calculate Gini coefficient
        cumsum = 0
        for i, value in enumerate(sorted_values):
            cumsum += (i + 1) * value
        
        # Normalize
        return (2 * cumsum) / (n * sum(sorted_values)) - (n + 1) / n

    def generate_visualizations(self):
        """Generate comprehensive visualizations from the analysis results."""
        print("Generating comprehensive visualizations...")
        
        # Create visualizations directory
        os.makedirs('visualizations', exist_ok=True)
        
        # Most of the visualizations are created in their respective analysis methods
        # This method adds a few more global visualizations
        
        # 1. PCA visualization of meetings based on topics
        print("Creating PCA visualization of meetings...")
        
        # Prepare document-term matrix
        meetings = []
        texts = []
        
        for filename, analysis in self.meeting_analyses.items():
            if 'open_ended_analysis' in analysis and 'primary_themes' in analysis['open_ended_analysis']:
                themes = [theme['theme'] for theme in analysis['open_ended_analysis']['primary_themes'] if isinstance(theme, dict) and 'theme' in theme]
                if themes:
                    meetings.append(filename)
                    texts.append(' '.join(themes))
        
        if texts:
            # Create a TF-IDF matrix
            vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
            X = vectorizer.fit_transform(texts)
            
            # Apply PCA for dimensionality reduction
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X.toarray())
            
            # Create scatter plot
            plt.figure(figsize=(12, 10))
            
            # Color points by year
            years = [self.transcripts[filename]['year'] for filename in meetings]
            unique_years = sorted(set(years))
            colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_years)))
            year_to_color = {year: colors[i] for i, year in enumerate(unique_years)}
            
            for i, (x, y) in enumerate(X_pca):
                year = years[i]
                plt.scatter(x, y, color=year_to_color[year], alpha=0.7)
            
            # Add a legend
            for year in unique_years:
                plt.scatter([], [], color=year_to_color[year], label=year)
            
            plt.title('PCA Visualization of Meetings Based on Topics')
            plt.xlabel('Component 1')
            plt.ylabel('Component 2')
            plt.legend(title='Year')
            plt.tight_layout()
            plt.savefig('visualizations/meeting_topics_pca.png', dpi=300)
            plt.close()
        
        # 2. Focus Areas Correlation
        print("Creating focus areas correlation visualization...")
        
        # Prepare correlation data
        focus_data = []
        
        for filename, analysis in self.meeting_analyses.items():
            if 'directed_analysis' in analysis:
                row = {
                    'filename': filename,
                    'date': self.transcripts[filename]['date'],
                    'land_use': 1 if (analysis['directed_analysis'].get('land_use', {}).get('present', False)) else 0,
                    'economic_development': 1 if (analysis['directed_analysis'].get('economic_development', {}).get('present', False)) else 0,
                    'taxes_budget': 1 if (analysis['directed_analysis'].get('taxes_budget', {}).get('present', False)) else 0
                }
                focus_data.append(row)
        
        if focus_data:
            focus_df = pd.DataFrame(focus_data)
            
            # Calculate correlation
            corr = focus_df[['land_use', 'economic_development', 'taxes_budget']].corr()
            
            # Create heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
            plt.title('Correlation Between Focus Areas')
            plt.tight_layout()
            plt.savefig('visualizations/focus_areas_correlation.png', dpi=300)
            plt.close()
        
        # 3. Top 20 Topics Overall
        print("Creating top topics visualization...")
        
        all_themes = Counter()
        
        for filename, analysis in self.meeting_analyses.items():
            if 'open_ended_analysis' in analysis and 'primary_themes' in analysis['open_ended_analysis']:
                for theme in analysis['open_ended_analysis']['primary_themes']:
                    if isinstance(theme, dict) and 'theme' in theme:
                        all_themes[theme['theme']] += 1
        
        if all_themes:
            top_themes = all_themes.most_common(20)
            
            plt.figure(figsize=(15, 10))
            themes, counts = zip(*top_themes)
            plt.barh(themes, counts)
            plt.xlabel('Number of Meetings')
            plt.title('Top 20 Topics Across All Meetings')
            plt.gca().invert_yaxis()  # Display highest counts at the top
            plt.tight_layout()
            plt.savefig('visualizations/top_topics_overall.png', dpi=300)
            plt.close()
        
        print("Comprehensive visualizations complete.")

    def generate_reports(self):
        """Generate comprehensive reports from the analysis results."""
        print("Generating comprehensive reports...")
        
        # Create reports directory
        os.makedirs('reports', exist_ok=True)
        
        # 1. Executive Summary Report
        with open('reports/executive_summary.md', 'w') as f:
            f.write("# Manhattan City Council Meetings Analysis: Executive Summary\n\n")
            
            f.write("## Overview\n\n")
            f.write(f"This report analyzes {len(self.meeting_analyses)} Manhattan, Kansas city council meetings from {min([data['date'] for _, data in self.transcripts.items() if data['date'] != 'Unknown'])} to {max([data['date'] for _, data in self.transcripts.items() if data['date'] != 'Unknown'])}.\n\n")
            
            f.write("## Key Findings\n\n")
            
            # Add key findings from global patterns
            if isinstance(self.global_patterns, dict) and 'error' not in self.global_patterns:
                for key, value in self.global_patterns.items():
                    if key.lower() in ['major_trends', 'key_findings', 'significant_patterns', 'unexpected_insights']:
                        f.write(f"### {key.replace('_', ' ').title()}\n\n")
                        
                        if isinstance(value, list):
                            for item in value:
                                if isinstance(item, dict) and 'finding' in item:
                                    f.write(f"- **{item.get('title', 'Finding')}**: {item['finding']}\n")
                                elif isinstance(item, str):
                                    f.write(f"- {item}\n")
                            f.write("\n")
                        elif isinstance(value, dict):
                            for subkey, subvalue in value.items():
                                f.write(f"- **{subkey.replace('_', ' ').title()}**: {subvalue}\n")
                            f.write("\n")
                        elif isinstance(value, str):
                            f.write(f"{value}\n\n")
            
            # Add findings on Dr. Lind's focus areas
            f.write("### Focus Area Analysis\n\n")
            
            # Count meetings where each focus area is present
            land_use_count = sum(1 for _, analysis in self.meeting_analyses.items() 
                              if 'directed_analysis' in analysis and 
                              'land_use' in analysis['directed_analysis'] and 
                              analysis['directed_analysis']['land_use'].get('present', False))
            
            econ_dev_count = sum(1 for _, analysis in self.meeting_analyses.items() 
                               if 'directed_analysis' in analysis and 
                               'economic_development' in analysis['directed_analysis'] and 
                               analysis['directed_analysis']['economic_development'].get('present', False))
            
            tax_budget_count = sum(1 for _, analysis in self.meeting_analyses.items() 
                                 if 'directed_analysis' in analysis and 
                                 'taxes_budget' in analysis['directed_analysis'] and 
                                 analysis['directed_analysis']['taxes_budget'].get('present', False))
            
            total_meetings = len(self.meeting_analyses)
            
            f.write(f"- **Land Use**: Present in {land_use_count} meetings ({land_use_count/total_meetings*100:.1f}% of total)\n")
            f.write(f"- **Economic Development**: Present in {econ_dev_count} meetings ({econ_dev_count/total_meetings*100:.1f}% of total)\n")
            f.write(f"- **Taxes and Budget**: Present in {tax_budget_count} meetings ({tax_budget_count/total_meetings*100:.1f}% of total)\n\n")
            
            # Add emerging themes
            f.write("### Emergent Themes\n\n")
            
            # Collect recurring themes
            all_recurring_themes = Counter()
            
            for filename, analysis in self.meeting_analyses.items():
                if 'emergent_patterns' in analysis and 'recurring_themes' in analysis['emergent_patterns']:
                    for theme in analysis['emergent_patterns']['recurring_themes']:
                        if isinstance(theme, dict) and 'theme' in theme:
                            all_recurring_themes[theme['theme']] += 1
            
            # List top recurring themes
            for theme, count in all_recurring_themes.most_common(10):
                f.write(f"- **{theme}** (appears in {count} meetings)\n")
            
            f.write("\n")
            
            # Add speaker patterns
            f.write("### Speaker Dynamics\n\n")
            
            # Calculate average speakers per meeting
            avg_speakers = sum(analysis.get('speaker_analysis', {}).get('speakers', {}).get('count', 0) 
                             for _, analysis in self.meeting_analyses.items()) / max(1, len(self.meeting_analyses))
            
            f.write(f"- Average of {avg_speakers:.1f} unique speakers per meeting\n")
            
            # Calculate average Gini coefficient
            gini_values = []
            for filename, analysis in self.meeting_analyses.items():
                if 'speaker_analysis' in analysis and 'speakers' in analysis['speaker_analysis'] and 'speaking_time' in analysis['speaker_analysis']['speakers']:
                    speaking_times = list(analysis['speaker_analysis']['speakers']['speaking_time'].values())
                    if speaking_times:
                        gini_values.append(self._calculate_gini(speaking_times))
            
            if gini_values:
                avg_gini = sum(gini_values) / len(gini_values)
                f.write(f"- Average speaking equality (Gini coefficient): {avg_gini:.3f} (0 = perfect equality, 1 = perfect inequality)\n")
            
            # Count dominant speakers
            dominant_speakers = Counter()
            for filename, analysis in self.meeting_analyses.items():
                if 'speaker_analysis' in analysis and 'power_dynamics' in analysis['speaker_analysis'] and 'dominant_speakers' in analysis['speaker_analysis']['power_dynamics']:
                    for speaker in analysis['speaker_analysis']['power_dynamics']['dominant_speakers']:
                        dominant_speakers[speaker] += 1
            
            if dominant_speakers:
                f.write("\n**Most Frequently Dominant Speakers:**\n\n")
                for speaker, count in dominant_speakers.most_common(5):
                    f.write(f"- {speaker}: Dominant in {count} meetings\n")
            
            f.write("\n")
            
            # Add anomalous meetings
            if self.anomalies:
                f.write("### Anomalous Meetings\n\n")
                
                for i, meeting in enumerate(self.anomalies[:5]):
                    f.write(f"**{i+1}. {meeting['date']} ({meeting['filename']}):**\n\n")
                    
                    for metric, zscore in meeting['anomaly_scores']:
                        metric_name = metric.replace('_zscore', '')
                        direction = "above" if zscore > 0 else "below"
                        f.write(f"- {metric_name.replace('_', ' ').title()}: {abs(zscore):.1f} standard deviations {direction} average\n")
                    
                    f.write("\n")
            
            f.write("## Conclusion\n\n")
            f.write("This executive summary highlights key patterns identified in the analysis of Manhattan city council meetings. For detailed findings, please refer to the specific reports on focus areas, speaker dynamics, temporal trends, and emergent patterns.\n\n")
            f.write("The analysis reveals both expected and unexpected patterns in how the council operates, what topics dominate discussions, and how participation and focus areas have evolved over time.\n")
        
        # 2. Focus Areas Deep Dive Report
        with open('reports/focus_areas_report.md', 'w') as f:
            f.write("# Focus Areas Analysis: Land Use, Economic Development, and Taxes/Budget\n\n")
            
            # Land Use section
            f.write("## Land Use Analysis\n\n")
            
            land_use_meetings = []
            for filename, analysis in self.meeting_analyses.items():
                if 'directed_analysis' in analysis and 'land_use' in analysis['directed_analysis'] and analysis['directed_analysis']['land_use'].get('present', False):
                    land_use_meetings.append({
                        'filename': filename,
                        'date': self.transcripts[filename]['date'],
                        'significance': analysis['directed_analysis']['land_use'].get('significance', 'Low'),
                        'context': analysis['directed_analysis']['land_use'].get('context', ''),
                        'notable_points': analysis['directed_analysis']['land_use'].get('notable_points', [])
                    })
            
            f.write(f"Land use topics were discussed in {len(land_use_meetings)} out of {len(self.meeting_analyses)} meetings ({len(land_use_meetings)/len(self.meeting_analyses)*100:.1f}%).\n\n")
            
            # Count by significance
            high_sig = sum(1 for m in land_use_meetings if m['significance'] == 'High')
            medium_sig = sum(1 for m in land_use_meetings if m['significance'] == 'Medium')
            low_sig = sum(1 for m in land_use_meetings if m['significance'] == 'Low')
            
            f.write(f"- High significance: {high_sig} meetings\n")
            f.write(f"- Medium significance: {medium_sig} meetings\n")
            f.write(f"- Low significance: {low_sig} meetings\n\n")
            
            f.write("### Key Contexts for Land Use Discussions\n\n")
            
            # Extract unique contexts from high significance meetings
            high_sig_contexts = [m['context'] for m in land_use_meetings if m['significance'] == 'High' and m['context']]
            for context in high_sig_contexts[:5]:
                f.write(f"- {context}\n")
            
            f.write("\n### Notable Points About Land Use\n\n")
            
            # Collect all notable points
            all_points = []
            for meeting in land_use_meetings:
                all_points.extend(meeting['notable_points'])
            
            # Count frequency of each point
            point_counter = Counter(all_points)
            
            # List most common points
            for point, count in point_counter.most_common(10):
                f.write(f"- {point} (mentioned in {count} meetings)\n")
            
            f.write("\n### Land Use Discussion Over Time\n\n")
            
            if 'temporal_trends' in self.__dict__ and 'focus_areas_by_year' in self.temporal_trends:
                years = self.temporal_trends['years']
                land_use_data = self.temporal_trends['focus_areas_by_year']['land_use']
                
                f.write("| Year | Meetings with Land Use | High Significance |\n")
                f.write("|------|------------------------|--------------------|\n")
                
                for year in years:
                    f.write(f"| {year} | {land_use_data[year]['count']} | {land_use_data[year]['high']} |\n")
            
            # Similar sections for Economic Development and Taxes/Budget
            # [Code omitted for brevity but would follow the same pattern]
        
        # 3. Emergent Patterns Report
        with open('reports/emergent_patterns_report.md', 'w') as f:
            f.write("# Emergent Patterns in Manhattan City Council Meetings\n\n")
            
            f.write("This report highlights patterns that emerged from the analysis beyond the predefined focus areas.\n\n")
            
            # Collect recurring themes
            all_recurring_themes = []
            for filename, analysis in self.meeting_analyses.items():
                if 'emergent_patterns' in analysis and 'recurring_themes' in analysis['emergent_patterns']:
                    for theme in analysis['emergent_patterns']['recurring_themes']:
                        if isinstance(theme, dict) and 'theme' in theme:
                            all_recurring_themes.append({
                                'theme': theme['theme'],
                                'evidence': theme.get('evidence', ''),
                                'meeting': filename,
                                'date': self.transcripts[filename]['date']
                            })
            
            # Group by theme
            theme_groups = {}
            for item in all_recurring_themes:
                theme = item['theme']
                if theme not in theme_groups:
                    theme_groups[theme] = []
                theme_groups[theme].append(item)
            
            # Sort by frequency
            sorted_themes = sorted(theme_groups.items(), key=lambda x: len(x[1]), reverse=True)
            
            f.write("## Recurring Themes\n\n")
            
            for theme, occurrences in sorted_themes[:10]:
                f.write(f"### {theme} (appears in {len(occurrences)} meetings)\n\n")
                
                # Show evidence from a few occurrences
                for occurrence in occurrences[:3]:
                    if occurrence['evidence']:
                        f.write(f"- **{occurrence['date']}**: {occurrence['evidence']}\n")
                
                f.write("\n")
            
            # Underlying tensions
            all_tensions = []
            for filename, analysis in self.meeting_analyses.items():
                if 'emergent_patterns' in analysis and 'underlying_tensions' in analysis['emergent_patterns']:
                    for tension in analysis['emergent_patterns']['underlying_tensions']:
                        if isinstance(tension, dict) and 'tension' in tension:
                            all_tensions.append({
                                'tension': tension['tension'],
                                'between': tension.get('between', []),
                                'context': tension.get('context', ''),
                                'meeting': filename,
                                'date': self.transcripts[filename]['date']
                            })
            
            # Group by tension
            tension_groups = {}
            for item in all_tensions:
                tension = item['tension']
                if tension not in tension_groups:
                    tension_groups[tension] = []
                tension_groups[tension].append(item)
            
            # Sort by frequency
            sorted_tensions = sorted(tension_groups.items(), key=lambda x: len(x[1]), reverse=True)
            
            f.write("## Underlying Tensions\n\n")
            
            for tension, occurrences in sorted_tensions[:10]:
                f.write(f"### {tension} (appears in {len(occurrences)} meetings)\n\n")
                
                # Show context from a few occurrences
                for occurrence in occurrences[:3]:
                    if occurrence['context']:
                        f.write(f"- **{occurrence['date']}**: {occurrence['context']}\n")
                    
                    if occurrence['between']:
                        f.write(f"  - Between: {', '.join(occurrence['between'])}\n")
                
                f.write("\n")
            
            # Decision patterns
            f.write("## Decision-Making Patterns\n\n")
            
            decision_patterns = Counter()
            for filename, analysis in self.meeting_analyses.items():
                if 'emergent_patterns' in analysis and 'decision_patterns' in analysis['emergent_patterns'] and 'pattern' in analysis['emergent_patterns']['decision_patterns']:
                    pattern = analysis['emergent_patterns']['decision_patterns']['pattern']
                    decision_patterns[pattern] += 1
            
            for pattern, count in decision_patterns.most_common(5):
                f.write(f"- **{pattern}** (observed in {count} meetings)\n")
            
            f.write("\n")
            
            # Language patterns
            f.write("## Language Patterns\n\n")
            
            formality_counter = Counter()
            tech_density_counter = Counter()
            emotional_tones = []
            
            for filename, analysis in self.meeting_analyses.items():
                if 'emergent_patterns' in analysis and 'language_patterns' in analysis['emergent_patterns']:
                    lang_patterns = analysis['emergent_patterns']['language_patterns']
                    
                    if 'formality' in lang_patterns:
                        formality_counter[lang_patterns['formality']] += 1
                    
                    if 'technical_density' in lang_patterns:
                        tech_density_counter[lang_patterns['technical_density']] += 1
                    
                    if 'emotional_tone' in lang_patterns and lang_patterns['emotional_tone']:
                        emotional_tones.append(lang_patterns['emotional_tone'])
            
            f.write("### Formality Level\n\n")
            for level, count in formality_counter.most_common():
                f.write(f"- {level}: {count} meetings ({count/len(self.meeting_analyses)*100:.1f}%)\n")
            
            f.write("\n### Technical Density\n\n")
            for level, count in tech_density_counter.most_common():
                f.write(f"- {level}: {count} meetings ({count/len(self.meeting_analyses)*100:.1f}%)\n")
            
            f.write("\n### Emotional Tone\n\n")
            
            # Create a word cloud of emotional tones if there are enough
            if len(emotional_tones) >= 5:
                # Join all tones into a single text
                all_tones = ' '.join(emotional_tones)
                
                # Create word cloud
                tone_cloud = WordCloud(width=800, height=400, background_color='white', max_words=50).generate(all_tones)
                
                # Save word cloud
                plt.figure(figsize=(10, 5))
                plt.imshow(tone_cloud, interpolation='bilinear')
                plt.axis('off')
                plt.tight_layout()
                plt.savefig('visualizations/emotional_tones_wordcloud.png')
                plt.close()
                
                f.write("A word cloud of emotional tones has been generated. See `visualizations/emotional_tones_wordcloud.png`.\n\n")
            
            # List a few common tones
            tone_counter = Counter(emotional_tones)
            for tone, count in tone_counter.most_common(5):
                f.write(f"- {tone} (observed in {count} meetings)\n")
            
            f.write("\n")
            
            # Other insights
            f.write("## Other Insights\n\n")
            
            all_insights = []
            for filename, analysis in self.meeting_analyses.items():
                if 'emergent_patterns' in analysis and 'other_insights' in analysis['emergent_patterns']:
                    for insight in analysis['emergent_patterns']['other_insights']:
                        if isinstance(insight, str) and insight:
                            all_insights.append((insight, filename))
            
            # Create a counter for insights
            insight_counter = Counter()
            for insight, _ in all_insights:
                insight_counter[insight] += 1
            
            # List most common insights
            for insight, count in insight_counter.most_common(10):
                f.write(f"- {insight} (observed in {count} meetings)\n")
        
        # 4. Speaker Dynamics Report
        with open('reports/speaker_dynamics_report.md', 'w') as f:
            f.write("# Speaker Dynamics in Manhattan City Council Meetings\n\n")
            
            f.write("This report analyzes patterns in speaker participation and interaction.\n\n")
            
            # Calculate average speakers per meeting
            avg_speakers = sum(analysis.get('speaker_analysis', {}).get('speakers', {}).get('count', 0) 
                             for _, analysis in self.meeting_analyses.items()) / max(1, len(self.meeting_analyses))
            
            f.write(f"## Overview\n\n")
            f.write(f"- Average of {avg_speakers:.1f} unique speakers per meeting\n")
            
            # Calculate average Gini coefficient
            gini_values = []
            for filename, analysis in self.meeting_analyses.items():
                if 'speaker_analysis' in analysis and 'speakers' in analysis['speaker_analysis'] and 'speaking_time' in analysis['speaker_analysis']['speakers']:
                    speaking_times = list(analysis['speaker_analysis']['speakers']['speaking_time'].values())
                    if speaking_times:
                        gini_values.append(self._calculate_gini(speaking_times))
            
            if gini_values:
                avg_gini = sum(gini_values) / len(gini_values)
                f.write(f"- Average speaking equality (Gini coefficient): {avg_gini:.3f} (0 = perfect equality, 1 = perfect inequality)\n\n")
            
            # Power dynamics
            f.write("## Power Dynamics\n\n")
            
            # Count dominant speakers
            dominant_speakers = Counter()
            for filename, analysis in self.meeting_analyses.items():
                if 'speaker_analysis' in analysis and 'power_dynamics' in analysis['speaker_analysis'] and 'dominant_speakers' in analysis['speaker_analysis']['power_dynamics']:
                    for speaker in analysis['speaker_analysis']['power_dynamics']['dominant_speakers']:
                        dominant_speakers[speaker] += 1
            
            if dominant_speakers:
                f.write("### Most Frequently Dominant Speakers\n\n")
                for speaker, count in dominant_speakers.most_common(10):
                    f.write(f"- **{speaker}**: Dominant in {count} meetings\n")
                
                f.write("\n")
            
            # Collect power dynamics observations
            power_observations = []
            for filename, analysis in self.meeting_analyses.items():
                if 'speaker_analysis' in analysis and 'power_dynamics' in analysis['speaker_analysis'] and 'observation' in analysis['speaker_analysis']['power_dynamics']:
                    observation = analysis['speaker_analysis']['power_dynamics']['observation']
                    if observation:
                        power_observations.append({
                            'observation': observation,
                            'meeting': filename,
                            'date': self.transcripts[filename]['date']
                        })
            
            if power_observations:
                f.write("### Key Observations on Power Dynamics\n\n")
                
                # Create a word cloud of power observations
                all_observations = ' '.join([o['observation'] for o in power_observations])
                
                # Create word cloud
                power_cloud = WordCloud(width=800, height=400, background_color='white', max_words=100).generate(all_observations)
                
                # Save word cloud
                plt.figure(figsize=(10, 5))
                plt.imshow(power_cloud, interpolation='bilinear')
                plt.axis('off')
                plt.tight_layout()
                plt.savefig('visualizations/power_dynamics_wordcloud.png')
                plt.close()
                
                f.write("A word cloud of power dynamics observations has been generated. See `visualizations/power_dynamics_wordcloud.png`.\n\n")
                
                # List a few notable observations
                f.write("#### Notable Observations:\n\n")
                for i, obs in enumerate(power_observations[:5]):
                    f.write(f"{i+1}. **{obs['date']}**: {obs['observation']}\n\n")
            
            # Interaction patterns
            f.write("## Interaction Patterns\n\n")
            
            # Collect interaction pattern observations
            interaction_observations = []
            for filename, analysis in self.meeting_analyses.items():
                if 'speaker_analysis' in analysis and 'interaction_patterns' in analysis['speaker_analysis'] and 'pattern' in analysis['speaker_analysis']['interaction_patterns']:
                    pattern = analysis['speaker_analysis']['interaction_patterns']['pattern']
                    if pattern:
                        interaction_observations.append({
                            'pattern': pattern,
                            'meeting': filename,
                            'date': self.transcripts[filename]['date']
                        })
            
            if interaction_observations:
                # Group similar patterns
                pattern_counter = Counter()
                for obs in interaction_observations:
                    pattern_counter[obs['pattern']] += 1
                
                f.write("### Common Interaction Patterns\n\n")
                for pattern, count in pattern_counter.most_common(5):
                    f.write(f"- **{pattern}** (observed in {count} meetings)\n")
                
                f.write("\n")
            
            # Notable exchanges
            all_exchanges = []
            for filename, analysis in self.meeting_analyses.items():
                if 'speaker_analysis' in analysis and 'interaction_patterns' in analysis['speaker_analysis'] and 'notable_exchanges' in analysis['speaker_analysis']['interaction_patterns']:
                    for exchange in analysis['speaker_analysis']['interaction_patterns']['notable_exchanges']:
                        if exchange:
                            all_exchanges.append({
                                'exchange': exchange,
                                'meeting': filename,
                                'date': self.transcripts[filename]['date']
                            })
            
            if all_exchanges:
                f.write("### Notable Exchanges\n\n")
                for i, exchange in enumerate(all_exchanges[:10]):
                    f.write(f"{i+1}. **{exchange['date']}**: {exchange['exchange']}\n\n")
            
            # Topic ownership
            f.write("## Topic Ownership\n\n")
            
            # Create a mapping of speaker to their topics
            speaker_topics = {}
            for filename, analysis in self.meeting_analyses.items():
                if 'speaker_analysis' in analysis and 'topic_ownership' in analysis['speaker_analysis']:
                    for item in analysis['speaker_analysis']['topic_ownership']:
                        if isinstance(item, dict) and 'speaker' in item and 'topics' in item:
                            speaker = item['speaker']
                            topics = item['topics']
                            
                            if speaker not in speaker_topics:
                                speaker_topics[speaker] = Counter()
                            
                            for topic in topics:
                                speaker_topics[speaker][topic] += 1
            
            if speaker_topics:
                # Sort speakers by number of topics
                sorted_speakers = sorted(speaker_topics.items(), key=lambda x: sum(x[1].values()), reverse=True)
                
                f.write("### Speakers and Their Topics\n\n")
                for speaker, topics in sorted_speakers[:10]:
                    f.write(f"**{speaker}**:\n")
                    for topic, count in topics.most_common(5):
                        f.write(f"- {topic} ({count} mentions)\n")
                    f.write("\n")
            
            # Public participation
            f.write("## Public Participation\n\n")
            
            # Count public participation levels
            participation_levels = Counter()
            for filename, analysis in self.meeting_analyses.items():
                if 'speaker_analysis' in analysis and 'public_participation' in analysis['speaker_analysis'] and 'level' in analysis['speaker_analysis']['public_participation']:
                    level = analysis['speaker_analysis']['public_participation']['level']
                    participation_levels[level] += 1
            
            if participation_levels:
                f.write("### Public Participation Levels\n\n")
                for level, count in participation_levels.most_common():
                    f.write(f"- {level}: {count} meetings ({count/len(self.meeting_analyses)*100:.1f}%)\n")
                
                f.write("\n")
            
            # Collect public participation observations
            participation_observations = []
            for filename, analysis in self.meeting_analyses.items():
                if 'speaker_analysis' in analysis and 'public_participation' in analysis['speaker_analysis'] and 'observation' in analysis['speaker_analysis']['public_participation']:
                    observation = analysis['speaker_analysis']['public_participation']['observation']
                    if observation:
                        participation_observations.append({
                            'observation': observation,
                            'meeting': filename,
                            'date': self.transcripts[filename]['date'],
                            'level': analysis['speaker_analysis']['public_participation'].get('level', 'Unknown')
                        })
            
            if participation_observations:
                f.write("### Key Observations on Public Participation\n\n")
                
                # Group by level
                for level in ['High', 'Medium', 'Low', 'Unknown']:
                    level_obs = [o for o in participation_observations if o['level'] == level]
                    if level_obs:
                        f.write(f"#### {level} Public Participation:\n\n")
                        for i, obs in enumerate(level_obs[:3]):
                            f.write(f"{i+1}. **{obs['date']}**: {obs['observation']}\n\n")
        
        # 5. Temporal Trends Report
        with open('reports/temporal_trends_report.md', 'w') as f:
            f.write("# Temporal Trends in Manhattan City Council Meetings\n\n")
            
            f.write("This report analyzes how topics and participation patterns have evolved over time.\n\n")
            
            # Skip if temporal trends haven't been calculated
            if 'temporal_trends' not in self.__dict__:
                f.write("Temporal trends analysis has not been performed.\n")
                return
            
            years = self.temporal_trends['years']
            
            f.write("## Focus Areas Over Time\n\n")
            
            # Create a table of focus areas by year
            f.write("### Land Use Topics\n\n")
            f.write("| Year | Meetings with Land Use | High Significance | Medium Significance | Low Significance |\n")
            f.write("|------|------------------------|-------------------|--------------------|-----------------|\n")
            
            for year in years:
                land_use_data = self.temporal_trends['focus_areas_by_year']['land_use'][year]
                f.write(f"| {year} | {land_use_data['count']} | {land_use_data['high']} | {land_use_data['medium']} | {land_use_data['low']} |\n")
            
            f.write("\n")
            
            # Similar tables for economic development and taxes/budget
            f.write("### Economic Development Topics\n\n")
            f.write("| Year | Meetings with Economic Dev | High Significance | Medium Significance | Low Significance |\n")
            f.write("|------|----------------------------|-------------------|--------------------|-----------------|\n")
            
            for year in years:
                econ_data = self.temporal_trends['focus_areas_by_year']['economic_development'][year]
                f.write(f"| {year} | {econ_data['count']} | {econ_data['high']} | {econ_data['medium']} | {econ_data['low']} |\n")
            
            f.write("\n")
            
            f.write("### Taxes and Budget Topics\n\n")
            f.write("| Year | Meetings with Taxes/Budget | High Significance | Medium Significance | Low Significance |\n")
            f.write("|------|------------------------------|-------------------|--------------------|-----------------|\n")
            
            for year in years:
                tax_data = self.temporal_trends['focus_areas_by_year']['taxes_budget'][year]
                f.write(f"| {year} | {tax_data['count']} | {tax_data['high']} | {tax_data['medium']} | {tax_data['low']} |\n")
            
            f.write("\n")
            
            # Top topics by year
            f.write("## Top Topics by Year\n\n")
            
            for year in years:
                if year in self.temporal_trends['topic_by_year'] and self.temporal_trends['topic_by_year'][year]:
                    f.write(f"### {year}\n\n")
                    
                    for topic, count in self.temporal_trends['topic_by_year'][year].most_common(5):
                        f.write(f"- {topic}: {count} meetings\n")
                    
                    f.write("\n")
            
            # Speaker equality over time
            f.write("## Speaker Equality Over Time\n\n")
            
            f.write("| Year | Average Gini Coefficient | Interpretation |\n")
            f.write("|------|--------------------------|----------------|\n")
            
            for year in years:
                if year in self.temporal_trends['speaking_equality_by_year'] and self.temporal_trends['speaking_equality_by_year'][year]:
                    avg_gini = sum(self.temporal_trends['speaking_equality_by_year'][year]) / len(self.temporal_trends['speaking_equality_by_year'][year])
                    
                    # Interpret the Gini coefficient
                    if avg_gini < 0.3:
                        interpretation = "Relatively equal speaking time"
                    elif avg_gini < 0.5:
                        interpretation = "Moderately unequal speaking time"
                    else:
                        interpretation = "Highly unequal speaking time"
                    
                    f.write(f"| {year} | {avg_gini:.3f} | {interpretation} |\n")
            
            f.write("\n")
            
            # Public participation over time
            f.write("## Public Participation Over Time\n\n")
            
            f.write("| Year | High | Medium | Low | Unknown |\n")
            f.write("|------|------|--------|-----|--------|\n")
            
            for year in years:
                if year in self.temporal_trends['public_participation_by_year']:
                    high = self.temporal_trends['public_participation_by_year'][year].get('High', 0)
                    medium = self.temporal_trends['public_participation_by_year'][year].get('Medium', 0)
                    low = self.temporal_trends['public_participation_by_year'][year].get('Low', 0)
                    unknown = self.temporal_trends['public_participation_by_year'][year].get('Unknown', 0)
                    
                    f.write(f"| {year} | {high} | {medium} | {low} | {unknown} |\n")
        
        # 6. Anomalous Meetings Report
        with open('reports/anomalous_meetings_report.md', 'w') as f:
            f.write("# Anomalous Manhattan City Council Meetings\n\n")
            
            f.write("This report highlights meetings that significantly deviate from typical patterns.\n\n")
            
            if not self.anomalies:
                f.write("No significant anomalies were detected in the meeting data.\n")
                return
            
            for i, meeting in enumerate(self.anomalies):
                f.write(f"## {i+1}. {meeting['date']} ({meeting['filename']})\n\n")
                
                f.write("### Anomaly Metrics\n\n")
                
                for metric, zscore in meeting['anomaly_scores']:
                    metric_name = metric.replace('_zscore', '')
                    direction = "above" if zscore > 0 else "below"
                    f.write(f"- **{metric_name.replace('_', ' ').title()}**: {abs(zscore):.1f} standard deviations {direction} average\n")
                
                f.write("\n")
                
                # Add details from the meeting analysis
                analysis = self.meeting_analyses.get(meeting['filename'], {})
                
                if 'open_ended_analysis' in analysis and 'primary_themes' in analysis['open_ended_analysis']:
                    f.write("### Primary Themes\n\n")
                    
                    for theme in analysis['open_ended_analysis']['primary_themes']:
                        if isinstance(theme, dict) and 'theme' in theme:
                            f.write(f"- **{theme['theme']}**")
                            if 'description' in theme and theme['description']:
                                f.write(f": {theme['description']}")
                            f.write("\n")
                    
                    f.write("\n")
                
                if 'emergent_patterns' in analysis and 'underlying_tensions' in analysis['emergent_patterns']:
                    f.write("### Underlying Tensions\n\n")
                    
                    for tension in analysis['emergent_patterns']['underlying_tensions']:
                        if isinstance(tension, dict) and 'tension' in tension:
                            f.write(f"- **{tension['tension']}**")
                            if 'context' in tension and tension['context']:
                                f.write(f": {tension['context']}")
                            f.write("\n")
                    
                    f.write("\n")
                
                if 'speaker_analysis' in analysis and 'power_dynamics' in analysis['speaker_analysis']:
                    f.write("### Power Dynamics\n\n")
                    
                    if 'dominant_speakers' in analysis['speaker_analysis']['power_dynamics']:
                        f.write("**Dominant Speakers**: " + ", ".join(analysis['speaker_analysis']['power_dynamics']['dominant_speakers']) + "\n\n")
                    
                    if 'observation' in analysis['speaker_analysis']['power_dynamics'] and analysis['speaker_analysis']['power_dynamics']['observation']:
                        f.write(f"**Observation**: {analysis['speaker_analysis']['power_dynamics']['observation']}\n\n")
                
                f.write("---\n\n")
        
        print("Reports generated successfully.")

    async def run_analysis(self):
        """Run the complete analysis pipeline."""
        # Step 1: Load transcripts
        self.load_transcripts()
        
        # Step 2: Analyze individual transcripts
        await self.analyze_all_transcripts()
        
        # Step 3: Detect global patterns
        await self.detect_global_patterns()
        
        # Step 4: Build topic and speaker networks
        self.build_topic_network()
        self.build_speaker_network()
        
        # Step 5: Analyze temporal trends
        self.analyze_temporal_trends()
        
        # Step 6: Detect anomalies
        self.detect_anomalies()
        
        # Step 7: Generate visualizations
        self.generate_visualizations()
        
        # Step 8: Generate reports
        self.generate_reports()
        
        print("Analysis pipeline complete!")
        return {
            'meeting_analyses': self.meeting_analyses,
            'global_patterns': self.global_patterns,
            'topic_network': self.topic_network,
            'speaker_network': self.speaker_network,
            'temporal_trends': self.temporal_trends,
            'anomalies': self.anomalies
        }

# Main execution
async def main():
    print("Starting analysis...")
    start_time = time.time()
    folder_path = r"C:\Python39\citymeeting"  # Update with your folder path
    
    analyzer = ComprehensiveMeetingAnalyzer(folder_path)
    results = await analyzer.run_analysis()
    
    end_time = time.time()
    print(f"Total analysis time: {(end_time - start_time) / 60:.2f} minutes")
    print("Analysis results, visualizations, and reports have been generated.")

if __name__ == "__main__":
    # Run the analysis
    asyncio.run(main())