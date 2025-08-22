import streamlit as st
import os
import warnings
from dotenv import load_dotenv

# Suppress specific warnings
warnings.filterwarnings("ignore", message=".*missing ScriptRunContext.*")
warnings.filterwarnings("ignore", message=".*USER_AGENT environment variable not set.*")

from langchain_community.document_loaders import WebBaseLoader
import validators
from langchain_community.document_loaders import YoutubeLoader
from groq import Groq

def init_session_state():
    """Initialize session state variables"""
    if 'llm' not in st.session_state:
        st.session_state.llm = None
    if 'model_name' not in st.session_state:
        st.session_state.model_name = None
    if 'model_initialized' not in st.session_state:
        st.session_state.model_initialized = False

def load_env_variables():
    """Load environment variables"""
    load_dotenv()
    os.environ["USER_AGENT"] = "streamlit-summary-app/1.0"
    
    groq_api_key = os.getenv("GROQ")
    if not groq_api_key:
        st.error("❌ GROQ not found in .env file. Please add it to your .env file.")
        st.stop()
    
    return groq_api_key

def initialize_model(groq_api_key):
    if st.session_state.model_initialized:
        return st.session_state.llm, st.session_state.model_name
    
    # Available Groq models - choose one that fits your needs
    model_name = "openai/gpt-oss-120b"  # Updated to a more reliable model
    
    try:
        # Initialize Groq client without model parameter
        llm = Groq(api_key=groq_api_key)
        
        st.session_state.llm = llm
        st.session_state.model_name = model_name
        st.session_state.model_initialized = True
        return llm, model_name
    except Exception as e:
        st.error(f"❌ Could not initialize Groq model: {str(e)}")
        st.stop()

def format_docs(documents):
    """Format documents into a single string"""
    return "\n\n".join([doc.page_content for doc in documents])

def is_youtube_url(url):
    """Check if URL is a valid YouTube URL"""
    youtube_patterns = [
        "youtube.com/watch",
        "youtu.be/",
        "youtube.com/embed/",
        "youtube.com/v/",
        "m.youtube.com/watch"
    ]
    return any(pattern in url.lower() for pattern in youtube_patterns)

def extract_video_id(url):
    """Extract video ID from YouTube URL"""
    import re
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/|youtube\.com\/v\/)([^&\n?#]+)',
        r'youtube\.com\/watch\?.*v=([^&\n?#]+)'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def get_youtube_transcript_direct(video_id):
    """Get YouTube transcript using direct API approach"""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        
        # Initialize the API
        ytt_api = YouTubeTranscriptApi()
        
        # Method 1: Try to fetch transcript directly
        try:
            transcript = ytt_api.fetch(video_id, languages=['en', 'en-US', 'en-GB'])
            # Handle both old dict format and new object format
            if hasattr(transcript[0], 'text'):
                # New format: objects with attributes
                formatted_content = "\n".join([item.text for item in transcript])
            else:
                # Old format: dictionaries
                formatted_content = "\n".join([item['text'] for item in transcript])
            return formatted_content, "Direct fetch (English)"
        except Exception as e1:
            st.warning(f"Direct English fetch failed: {str(e1)[:100]}...")
            
            # Method 2: Try to list available transcripts first
            try:
                transcript_list = ytt_api.list(video_id)
                
                # Try to find English transcript
                try:
                    transcript_obj = transcript_list.find_transcript(['en', 'en-US', 'en-GB'])
                    transcript_data = transcript_obj.fetch()
                    # Handle both formats
                    if hasattr(transcript_data[0], 'text'):
                        formatted_content = "\n".join([item.text for item in transcript_data])
                    else:
                        formatted_content = "\n".join([item['text'] for item in transcript_data])
                    return formatted_content, f"Found transcript ({transcript_obj.language})"
                except Exception as e_inner:
                    st.warning(f"English transcript search failed: {str(e_inner)[:100]}...")
                    
                    # Try manually created transcripts
                    try:
                        transcript_obj = transcript_list.find_manually_created_transcript(['en', 'en-US', 'en-GB'])
                        transcript_data = transcript_obj.fetch()
                        if hasattr(transcript_data[0], 'text'):
                            formatted_content = "\n".join([item.text for item in transcript_data])
                        else:
                            formatted_content = "\n".join([item['text'] for item in transcript_data])
                        return formatted_content, f"Manual transcript ({transcript_obj.language})"
                    except Exception as e_manual:
                        st.warning(f"Manual transcript failed: {str(e_manual)[:100]}...")
                        
                        # Try any generated transcript
                        try:
                            transcript_obj = transcript_list.find_generated_transcript(['en', 'en-US', 'en-GB'])
                            transcript_data = transcript_obj.fetch()
                            if hasattr(transcript_data[0], 'text'):
                                formatted_content = "\n".join([item.text for item in transcript_data])
                            else:
                                formatted_content = "\n".join([item['text'] for item in transcript_data])
                            return formatted_content, f"Generated transcript ({transcript_obj.language})"
                        except Exception as e_generated:
                            st.warning(f"Generated transcript failed: {str(e_generated)[:100]}...")
                            
                            # Get first available transcript in any language
                            try:
                                available_transcripts = list(transcript_list)
                                if available_transcripts:
                                    transcript_obj = available_transcripts[0]
                                    transcript_data = transcript_obj.fetch()
                                    if hasattr(transcript_data[0], 'text'):
                                        formatted_content = "\n".join([item.text for item in transcript_data])
                                    else:
                                        formatted_content = "\n".join([item['text'] for item in transcript_data])
                                    return formatted_content, f"Available transcript ({transcript_obj.language})"
                                else:
                                    raise Exception("No transcripts available")
                            except Exception as e_any:
                                raise Exception(f"No available transcripts: {str(e_any)}")
            except Exception as e2:
                raise Exception(f"Could not access transcripts: {str(e2)}")
                
    except ImportError:
        raise Exception("youtube-transcript-api not installed. Please install it with: pip install youtube-transcript-api")
    except Exception as e:
        raise Exception(f"Transcript API error: {str(e)}")

def estimate_tokens(text):
    """Rough estimation of tokens (1 token ≈ 4 characters for English)"""
    return len(text) // 4

def dynamic_chunk_text(text, content_type="article"):
    """
    Dynamically chunk text based on content length and type.
    Uses smaller, more manageable chunks for better processing.
    """
    # Estimate total tokens
    est_tokens = estimate_tokens(text)
    
    # Decide chunk size based on content length and type
    if content_type == "video":
        # YouTube videos often have repetitive content and need smaller chunks
        if est_tokens <= 5000:
            max_tokens = 1800  # Small videos - larger chunks
        elif est_tokens <= 15000:
            max_tokens = 2200  # Medium videos
        elif est_tokens <= 30000:
            max_tokens = 2000  # Large videos
        else:
            max_tokens = 1600  # Very large videos - smaller chunks for better processing
    else:
        # Articles and web content
        if est_tokens <= 5000:
            max_tokens = 2500
        elif est_tokens <= 20000:
            max_tokens = 3000
        else:
            max_tokens = 2200
    
    st.info(f"📊 Content analysis: {est_tokens:,} tokens estimated, using {max_tokens} token chunks")
    
    # Convert tokens to approximate characters
    max_chars = max_tokens * 4
    
    # Try to split by sentences first for better coherence
    sentences = text.split('. ')
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        test_chunk = current_chunk + sentence + ". "
        if len(test_chunk) <= max_chars:
            current_chunk = test_chunk
        else:
            # If current chunk has content, save it
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            
            # Start new chunk with current sentence
            current_chunk = sentence + ". "
            
            # If single sentence is too long, split it by character limit
            if len(current_chunk) > max_chars:
                # Split the long sentence
                long_sentence = current_chunk
                chunks.extend([long_sentence[i:i+max_chars] for i in range(0, len(long_sentence), max_chars)])
                current_chunk = ""
    
    # Add the last chunk if it has content
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    # Filter out very small chunks (less than 100 characters)
    chunks = [chunk for chunk in chunks if len(chunk.strip()) >= 100]
    
    return chunks

def chunk_text(text, max_chunk_size=3000):
    """Legacy chunking function - kept for backward compatibility"""
    return dynamic_chunk_text(text, "article")

def summarize_chunk(llm, chunk, summary_length, content_type, temperature, chunk_num=1, total_chunks=1):
    """Summarize a single chunk with improved prompting"""
    length_instructions = {
        "short": "Write a concise summary in 2-3 sentences focusing on the key points.",
        "medium": "Write a clear summary in 1-2 paragraphs covering the main ideas and important details.",
        "long": "Write a comprehensive summary in 2-3 paragraphs including key points, supporting details, and context.",
        "Very Long": "Write a detailed summary in 3-4 paragraphs with comprehensive coverage of all important information."
    }
    
    # Adjust approach based on content type
    if content_type == "video":
        context_instruction = "This is part of a video transcript. Focus on the main topics discussed, key insights, and important information presented."
    else:
        context_instruction = "Focus on the main arguments, key information, and important details."
    
    chunk_prompt = f"""Please summarize this content (part {chunk_num} of {total_chunks}).

{length_instructions[summary_length]}

{context_instruction}

Content:
{chunk}

Provide a clear, well-structured summary that captures the essential information from this section."""
    
    try:
        chat_completion = llm.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": chunk_prompt
                }
            ],
            model=st.session_state.model_name,
            temperature=temperature,
            max_tokens=800,  # Reduced for better chunk summaries
        )
        
        return chat_completion.choices[0].message.content
    except Exception as e:
        st.error(f"❌ Error summarizing chunk {chunk_num}: {str(e)}")
        return None

def combine_summaries(llm, chunk_summaries, summary_length, content_type, temperature):
    """Combine individual chunk summaries into a final coherent summary"""
    length_instructions = {
        "short": "Write a brief, coherent summary in 3-4 sentences.",
        "medium": "Write a comprehensive summary in 2-3 well-structured paragraphs.",
        "long": "Write a detailed summary in 4-6 paragraphs covering all major points and insights.",
        "Very Long": "Write an extensive summary in 6-10 paragraphs with comprehensive coverage of all topics and details."
    }
    
    combined_text = "\n\n".join([f"Section {i+1}: {summary}" for i, summary in enumerate(chunk_summaries)])
    
    content_context = {
        "video": "video content",
        "article": "article",
        "text": "document"
    }
    
    final_prompt = f"""Based on these section summaries, create a final unified summary of the entire {content_context.get(content_type, 'content')}.

{length_instructions[summary_length]}

Please create a coherent, well-organized summary that:
- Combines all key points from the sections
- Maintains logical flow and structure
- Avoids repetition
- Provides a complete overview of the content

Section Summaries:
{combined_text}

Create a unified summary that reads as a single, coherent piece rather than separate sections."""
    
    try:
        chat_completion = llm.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": final_prompt
                }
            ],
            model=st.session_state.model_name,
            temperature=temperature,
            max_tokens=2048,
        )
        
        return chat_completion.choices[0].message.content
    except Exception as e:
        st.error(f"❌ Error combining summaries: {str(e)}")
        return None

def summarize_content(llm, content, summary_length="medium", content_type="article", temperature=0.1):
    """Summarize content with intelligent dynamic chunking for large documents"""
    
    # Check if content needs chunking (use more conservative limit)
    max_safe_tokens = 3500  # Reduced for more reliable processing
    estimated_tokens = estimate_tokens(content)
    
    if estimated_tokens <= max_safe_tokens:
        # Small content - summarize directly
        length_instructions = {
            "short": "Write a brief summary in 3-4 sentences.",
            "medium": "Write a comprehensive summary in 2-3 paragraphs.",
            "long": "Write a comprehensive and detailed summary in 4-6 paragraphs. Include the main arguments, key points, supporting evidence, important details, conclusions, and any notable insights or implications.",
            "Very Long": "Write a comprehensive and detailed summary in 6-10 paragraphs. Include the main arguments, key points, supporting evidence, important details, conclusions, and any notable insights or implications."
        }
        
        content_prompts = {
            "article": "Article content:",
            "video": "Video transcript:",
            "text": "Text content:"
        }
        
        prompt_text = f"""{length_instructions[summary_length]}

{content_prompts.get(content_type, "Content:")}
{content}

Please provide a clear, well-structured summary that captures the main points and key insights."""
        
        try:
            chat_completion = llm.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt_text
                    }
                ],
                model=st.session_state.model_name,
                temperature=temperature,
                max_tokens=2048,
            )
            
            return chat_completion.choices[0].message.content
        except Exception as e:
            st.error(f"❌ Error generating summary: {str(e)}")
            return None
    
    else:
        # Large content - use dynamic chunking approach
        st.info(f"📊 Large content detected ({estimated_tokens:,} tokens). Using dynamic chunking...")
        
        # Split content into chunks using dynamic chunking
        chunks = dynamic_chunk_text(content, content_type)
        st.info(f"🔄 Processing {len(chunks)} optimized chunks...")
        
        # Show chunk size distribution
        chunk_sizes = [len(chunk) for chunk in chunks]
        avg_size = sum(chunk_sizes) / len(chunk_sizes)
        st.info(f"📈 Chunk info: Average size {avg_size:.0f} chars, Range: {min(chunk_sizes)}-{max(chunk_sizes)} chars")
        
        # Summarize each chunk
        chunk_summaries = []
        progress_bar = st.progress(0)
        
        for i, chunk in enumerate(chunks):
            with st.spinner(f"📝 Summarizing chunk {i+1} of {len(chunks)}... ({len(chunk):,} chars)"):
                summary = summarize_chunk(llm, chunk, summary_length, content_type, temperature, i+1, len(chunks))
                if summary:
                    chunk_summaries.append(summary)
                else:
                    st.warning(f"⚠️ Failed to summarize chunk {i+1}, skipping...")
                progress_bar.progress((i + 1) / len(chunks))
        
        if not chunk_summaries:
            st.error("❌ Failed to summarize any chunks")
            return None
        
        if len(chunk_summaries) < len(chunks):
            st.warning(f"⚠️ Only {len(chunk_summaries)} of {len(chunks)} chunks were successfully summarized")
        
        # Combine chunk summaries
        with st.spinner("🔄 Combining summaries into final result..."):
            final_summary = combine_summaries(llm, chunk_summaries, summary_length, content_type, temperature)
            
            if final_summary:
                st.success(f"✅ Successfully processed {len(chunks)} chunks into final summary!")
                st.info(f"📊 Processing stats: {len(chunk_summaries)} summaries combined from {estimated_tokens:,} tokens")
            
            return final_summary

def main():
    # Page configuration
    st.set_page_config(
        page_title="Document Summarizer",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    init_session_state()
    
    # Title and description
    st.title("📄 Document Summarizer")
    st.markdown("**Summarize web articles, documents, and YouTube videos using Groq AI with dynamic chunking**")
    
    # Load environment variables
    groq_api_key = load_env_variables()
    
    # Initialize model
    if not st.session_state.model_initialized:
        with st.spinner("🔄 Initializing AI model..."):
            llm, model_name = initialize_model(groq_api_key)
        st.success(f"✅ Successfully initialized model: {model_name}")
    else:
        llm = st.session_state.llm
        model_name = st.session_state.model_name
        st.success(f"✅ Using model: {model_name}")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Summary length option
    summary_length = st.sidebar.selectbox(
        "Summary Length",
        ["short", "medium", "long","Very Long"],
        index=1,
        help="Choose how detailed you want the summary to be"
    )
    
    # Temperature setting
    temperature = st.sidebar.slider(
        "Creativity (Temperature)",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.1,
        help="Higher values make output more creative but less predictable"
    )
    
    # Main content area
    st.header("📝 Input")
    
    # Tabs for different input methods
    tab1, tab2, tab3 = st.tabs(["🌐 Web URL", "📄 Text Input", "🎥 YouTube Video"])
    
    with tab1:
        st.subheader("Enter a Web URL")
        url_input = st.text_input(
            "URL",
            placeholder="https://example.com/article",
            help="Enter the URL of the article or webpage you want to summarize"
        )
        
        url_submit = st.button("📥 Load and Summarize URL", type="primary", use_container_width=True)
        
        if url_submit and url_input:
            if not validators.url(url_input):
                st.error("❌ Please enter a valid URL")
            else:
                with st.spinner("📖 Loading content from URL..."):
                    try:
                        # Load documents from URL
                        loader = WebBaseLoader(url_input)
                        docs = loader.load()
                        
                        if not docs:
                            st.error("❌ Could not load content from the URL")
                        else:
                            formatted_content = format_docs(docs)
                            
                            if len(formatted_content.strip()) < 50:
                                st.error("❌ The loaded content is too short to summarize.")
                            else:
                                # Generate summary
                                with st.spinner("✨ Generating summary..."):
                                    summary = summarize_content(llm, formatted_content, summary_length, "article", temperature)
                                    
                                    if summary:
                                        st.header("📋 Summary")
                                        st.write(summary)
                    
                    except Exception as e:
                        st.error(f"❌ Error loading URL: {str(e)}")
    
    with tab2:
        st.subheader("Enter Text Directly")
        text_input = st.text_area(
            "Text Content",
            placeholder="Paste your article or document text here...",
            height=300,
            help="Enter the text you want to summarize"
        )
        
        if text_input:
            st.info(f"📊 Text length: {len(text_input):,} characters (~{estimate_tokens(text_input):,} tokens)")
        
        text_submit = st.button("✨ Summarize Text", type="primary", use_container_width=True)
        
        if text_submit and text_input:
            if len(text_input.strip()) < 50:
                st.warning("⚠️ Please enter more text (at least 50 characters)")
            else:
                with st.spinner("✨ Generating summary..."):
                    summary = summarize_content(llm, text_input, summary_length, "text", temperature)
                    
                    if summary:
                        st.header("📋 Summary")
                        st.write(summary)
                        
                        # Copy-friendly output
                        with st.expander("📋 Copy Summary", expanded=False):
                            st.text_area(
                                "Summary text (copy-friendly)",
                                summary,
                                height=150,
                                disabled=True
                            )
    
    with tab3:
        st.subheader("Enter a YouTube Video URL")
        youtube_url = st.text_input(
            "YouTube URL",
            placeholder="https://www.youtube.com/watch?v=example or https://youtu.be/example",
            help="Enter a YouTube video link to summarize its transcript"
        )
        
        # Show example formats
        with st.expander("ℹ️ Supported YouTube URL formats", expanded=False):
            st.markdown("""
            - `https://www.youtube.com/watch?v=VIDEO_ID`
            - `https://youtu.be/VIDEO_ID`
            - `https://m.youtube.com/watch?v=VIDEO_ID`
            - `https://www.youtube.com/embed/VIDEO_ID`
            """)
        
        yt_submit = st.button("🎥 Load and Summarize Video", type="primary", use_container_width=True)
        
        if yt_submit and youtube_url:
            if not validators.url(youtube_url) or not is_youtube_url(youtube_url):
                st.error("❌ Please enter a valid YouTube URL")
            else:
                # Extract video ID
                video_id = extract_video_id(youtube_url)
                if not video_id:
                    st.error("❌ Could not extract video ID from URL")
                else:
                    st.info(f"🎯 Video ID: {video_id}")
                    
                    with st.spinner("📖 Loading transcript from YouTube..."):
                        success = False
                        formatted_content = None
                        method_used = None
                        
                        # Method 1: Try direct YouTube Transcript API with correct syntax
                        try:
                            st.info("🔄 Method 1: Using YouTube Transcript API directly...")
                            formatted_content, method_used = get_youtube_transcript_direct(video_id)
                            success = True
                            st.success(f"✅ Transcript loaded using direct API! ({method_used})")
                            
                        except Exception as e1:
                            st.warning(f"Method 1 failed: {str(e1)[:150]}...")
                            
                            # Method 2: Try LangChain YoutubeLoader
                            try:
                                st.info("🔄 Method 2: Using LangChain YoutubeLoader...")
                                loader = YoutubeLoader.from_youtube_url(youtube_url, add_video_info=True)
                                docs = loader.load()
                                if docs and docs[0].page_content.strip():
                                    formatted_content = format_docs(docs)
                                    success = True
                                    method_used = "LangChain YoutubeLoader"
                                    st.success("✅ Transcript loaded using LangChain!")
                            except Exception as e2:
                                st.warning(f"Method 2 failed: {str(e2)[:150]}...")
                                
                                # Method 3: Try LangChain with video ID only
                                try:
                                    st.info("🔄 Method 3: LangChain with video ID...")
                                    loader = YoutubeLoader(video_id)
                                    docs = loader.load()
                                    if docs and docs[0].page_content.strip():
                                        formatted_content = format_docs(docs)
                                        success = True
                                        method_used = "LangChain Video ID"
                                        st.success("✅ Transcript loaded with video ID method!")
                                except Exception as e3:
                                    # All methods failed
                                    error_msg = str(e3)
                                    if "400" in error_msg or "Bad Request" in error_msg:
                                        st.error("❌ **YouTube API Error**: The video might be:")
                                        st.markdown("""
                                        - 🔒 **Private or restricted**
                                        - 🚫 **Age-restricted** 
                                        - 🌍 **Region-blocked**
                                        - 📺 **Live stream** (transcripts not available for live content)
                                        - 🎵 **Music video** (often blocked)
                                        """)
                                    elif "No transcripts" in error_msg or "Transcript not available" in error_msg or "not available" in error_msg.lower():
                                        st.error("❌ **No Transcript Available**: This video doesn't have captions or subtitles.")
                                    elif "Video unavailable" in error_msg:
                                        st.error("❌ **Video Unavailable**: The video might be deleted or private.")
                                    elif "Invalid video" in error_msg or "not found" in error_msg:
                                        st.error("❌ **Invalid Video**: Check if the URL is correct.")
                                    else:
                                        st.error(f"❌ **Error**: {error_msg}")
                                    
                                    # Provide helpful suggestions
                                    st.info("""
                                    💡 **Suggestions:**
                                    - Try educational or tutorial videos (Khan Academy, Coursera)
                                    - Look for videos with the CC (closed captions) button
                                    - Avoid music videos or copyrighted content
                                    - Check if the video is public and not age-restricted
                                    - Try TED talks or public lectures
                                    """)
                                    
                                    # Show example of working video types
                                    with st.expander("✅ Examples of videos that usually work", expanded=False):
                                        st.markdown("""
                                        **Educational Content:**
                                        - Khan Academy lessons
                                        - Coursera course videos
                                        - University lectures
                                        - Programming tutorials
                                        
                                        **News & Talks:**
                                        - TED Talks
                                        - News broadcasts
                                        - Conference presentations
                                        - Product demos
                                        """)
                        
                        if success and formatted_content:
                            if len(formatted_content.strip()) < 50:
                                st.error("❌ Transcript is too short to summarize.")
                            else:
                                st.info(f"📋 **Method used**: {method_used}")
                                
                                # Show enhanced transcript info
                                est_tokens = estimate_tokens(formatted_content)
                                st.info(f"📊 Transcript: {len(formatted_content):,} characters (~{est_tokens:,} tokens)")
                                
                                # Show transcript preview
                                with st.expander("📖 Transcript Preview", expanded=False):
                                    preview_text = formatted_content[:1000] + "..." if len(formatted_content) > 1000 else formatted_content
                                    st.text_area(
                                        "Transcript",
                                        preview_text,
                                        height=200,
                                        disabled=True
                                    )
                                
                                # Generate summary with video-optimized processing
                                with st.spinner("✨ Generating video summary with optimized chunking..."):
                                    summary = summarize_content(llm, formatted_content, summary_length, "video", temperature)
                                    
                                    if summary:
                                        st.header("📋 Video Summary")
                                        st.write(summary)
                                        
                                        # Copy-friendly output
                                        with st.expander("📋 Copy Summary", expanded=False):
                                            st.text_area(
                                                "Summary text (copy-friendly)",
                                                summary,
                                                height=150,
                                                disabled=True
                                            )
    
    # Sidebar info
    st.sidebar.header("ℹ️ About")
    st.sidebar.info(
        """
        This app uses Groq AI with dynamic chunking to summarize content from multiple sources.
        
        **Features:**
        - 🌐 Web articles and blogs
        - 📄 Direct text input
        - 🎥 YouTube video transcripts
        - 📏 Adjustable summary length
        - 🎨 Customizable AI creativity
        - 🔄 Dynamic chunking for large content
        
        **Improvements:**
        - ⚡ Optimized chunk sizes for different content types
        - 🎯 Better handling of long YouTube videos
        - 📊 Smart token estimation and processing
        - 🔧 Enhanced error handling and recovery
        
        **Tips:**
        - Use **short** summaries for quick overviews
        - Use **long** summaries for detailed analysis
        - Use **very long** summaries for comprehensive coverage
        - YouTube videos need captions/transcripts
        - Adjust **creativity** for different writing styles
        """
    )
    
    # Model info
    if st.session_state.model_initialized:
        st.sidebar.success(f"🤖 Model: {st.session_state.model_name}")
    
    # Processing info
    st.sidebar.header("🔧 Processing Info")
    st.sidebar.info(
        """
        **Dynamic Chunking:**
        - Small content (≤5K tokens): Larger chunks
        - Medium content (≤15K tokens): Balanced chunks  
        - Large content (≤30K tokens): Optimized chunks
        - Very large content (>30K tokens): Smaller chunks
        
        **Video Processing:**
        - Specialized chunking for transcripts
        - Handles repetitive content better
        - Optimized for long-form videos
        """
    )
    
    # Dependencies check
    st.sidebar.header("📦 Dependencies")
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        st.sidebar.success("✅ YouTube Transcript API")
    except ImportError:
        st.sidebar.error("❌ YouTube Transcript API")
        st.sidebar.code("pip install youtube-transcript-api")
    
    # Performance tips
    st.sidebar.header("⚡ Performance Tips")
    st.sidebar.info(
        """
        **For Best Results:**
        - Content with clear structure works best
        - Educational videos perform better than entertainment
        - Articles with headings are processed more efficiently
        - Very long content (>50K tokens) may take longer
        """
    )
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("*Enhanced with dynamic chunking for better processing*")

if __name__ == "__main__":
    main()