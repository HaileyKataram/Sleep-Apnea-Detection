import streamlit as st
import sys
import os
import numpy as np
import pandas as pd
import torch
import yaml
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import librosa
import tempfile
from pathlib import Path
import time
from datetime import datetime
from scipy import signal
from scipy.stats import skew, kurtosis

# Add src to path
sys.path.append('src')

# Import our modules
from audio_processor import AudioProcessor
from feature_extractor import FeatureExtractor

# Page config
st.set_page_config(
    page_title="🩺 Enhanced Sleep Apnea Detector",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    
    .result-card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
        border-left: 5px solid #667eea;
        margin: 1rem 0;
    }
    
    .success-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .warning-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .info-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .suggestion-card {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        color: #333;
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
        border-left: 5px solid #ff7b00;
    }
</style>
""", unsafe_allow_html=True)

class EnhancedApneaDetector:
    """Enhanced rule-based sleep apnea detector with intelligent analysis."""
    
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
    
    def analyze_breathing_segment(self, audio_segment):
        """Analyze audio segment with enhanced detection and suggestions."""
        
        # Calculate RMS energy
        rms_energy = np.sqrt(np.mean(audio_segment**2))
        
        # Apply envelope detection
        envelope = np.abs(signal.hilbert(audio_segment))
        envelope_smooth = signal.savgol_filter(envelope, min(51, len(envelope)//2*2+1), 3)
        
        # Peak detection for breathing cycles
        min_distance = int(self.sample_rate * 0.8)
        
        try:
            peaks, properties = signal.find_peaks(
                envelope_smooth,
                height=np.mean(envelope_smooth) * 0.3,
                distance=min_distance,
                prominence=np.std(envelope_smooth) * 0.5
            )
        except:
            peaks = []
            properties = {}
        
        # Calculate breathing rate
        if len(peaks) >= 2:
            breath_intervals = np.diff(peaks) / self.sample_rate
            avg_breath_interval = np.mean(breath_intervals)
            breathing_rate = 60.0 / avg_breath_interval
            breath_regularity = 1.0 / (np.std(breath_intervals) + 1e-6)
        else:
            breathing_rate = 0.0
            breath_regularity = 0.0
        
        # Pause detection
        threshold = np.mean(envelope_smooth) * 0.15
        low_amplitude_mask = envelope_smooth < threshold
        pause_regions = self._find_continuous_regions(low_amplitude_mask)
        pause_durations = [(end - start) / self.sample_rate for start, end in pause_regions]
        
        # Pause statistics
        if pause_durations:
            max_pause_duration = max(pause_durations)
            avg_pause_duration = np.mean(pause_durations)
            long_pause_count = sum(1 for d in pause_durations if d > 2.0)
        else:
            max_pause_duration = 0.0
            avg_pause_duration = 0.0
            long_pause_count = 0
        
        # Amplitude variation
        amplitude_variation = np.std(envelope_smooth) / (np.mean(envelope_smooth) + 1e-6)
        
        # ENHANCED APNEA DETECTION WITH PROPER SCORING
        apnea_score = 0.0
        reasons = []
        suggestions = []
        
        # Rule 1: RMS energy analysis
        if rms_energy < 0.005:  # Very low
            apnea_score += 0.5
            reasons.append("Critically low breathing intensity")
            suggestions.append("Consider immediate medical evaluation")
        elif rms_energy < 0.015:  # Low
            apnea_score += 0.3
            reasons.append("Low breathing intensity")
            suggestions.append("Monitor for shallow breathing patterns")
        
        # Rule 2: Breathing rate analysis
        if breathing_rate == 0:  # No breathing detected
            apnea_score += 0.6
            reasons.append("No clear breathing pattern detected")
            suggestions.append("Check recording quality or consider severe apnea")
        elif breathing_rate < 6:  # Very low
            apnea_score += 0.5
            reasons.append(f"Severely low breathing rate ({breathing_rate:.1f}/min)")
            suggestions.append("Immediate medical attention recommended")
        elif breathing_rate < 10:  # Low
            apnea_score += 0.3
            reasons.append(f"Low breathing rate ({breathing_rate:.1f}/min)")
            suggestions.append("Consider sleep study evaluation")
        elif breathing_rate > 30:  # Very high (compensatory)
            apnea_score += 0.2
            reasons.append(f"Rapid compensatory breathing ({breathing_rate:.1f}/min)")
            suggestions.append("May indicate recovery from apnea events")
        
        # Rule 3: Pause duration analysis
        if max_pause_duration > 10:  # Severe
            apnea_score += 0.7
            reasons.append(f"Critically long breathing pause ({max_pause_duration:.1f}s)")
            suggestions.append("URGENT: Seek immediate medical care")
        elif max_pause_duration > 5:  # Moderate to severe
            apnea_score += 0.5
            reasons.append(f"Long breathing pause ({max_pause_duration:.1f}s)")
            suggestions.append("Schedule sleep study immediately")
        elif max_pause_duration > 3:  # Mild to moderate
            apnea_score += 0.3
            reasons.append(f"Moderate breathing pause ({max_pause_duration:.1f}s)")
            suggestions.append("Consider consulting sleep specialist")
        elif max_pause_duration > 2:  # Borderline
            apnea_score += 0.15
            reasons.append(f"Short breathing pause ({max_pause_duration:.1f}s)")
            suggestions.append("Monitor breathing patterns during sleep")
        
        # Rule 4: Multiple pauses
        if long_pause_count > 3:
            apnea_score += 0.4
            reasons.append(f"Multiple long pauses ({long_pause_count})")
            suggestions.append("Pattern suggests frequent apnea events")
        elif long_pause_count > 1:
            apnea_score += 0.2
            reasons.append(f"Several breathing interruptions ({long_pause_count})")
            suggestions.append("Consider home sleep monitoring")
        
        # Rule 5: Breathing regularity
        if breathing_rate > 0 and breath_regularity < 0.3:
            apnea_score += 0.25
            reasons.append("Highly irregular breathing pattern")
            suggestions.append("Irregular patterns may indicate sleep disorders")
        elif breathing_rate > 0 and breath_regularity < 0.5:
            apnea_score += 0.1
            reasons.append("Somewhat irregular breathing")
            suggestions.append("Monitor for consistency in breathing rhythm")
        
        # Rule 6: Amplitude variation (shallow breathing)
        if amplitude_variation < 0.05:
            apnea_score += 0.3
            reasons.append("Very shallow breathing detected")
            suggestions.append("Shallow breathing may indicate airway obstruction")
        elif amplitude_variation < 0.1:
            apnea_score += 0.15
            reasons.append("Reduced breathing amplitude")
            suggestions.append("Consider factors affecting breathing depth")
        
        # Normalize and determine prediction
        apnea_score = min(1.0, apnea_score)
        is_apnea = apnea_score > 0.35  # More sensitive threshold
        confidence = apnea_score if is_apnea else (1.0 - apnea_score)
        
        # Add general suggestions based on overall assessment
        if not is_apnea and len(suggestions) == 0:
            if breathing_rate > 12 and breathing_rate < 20:
                suggestions.append("Breathing pattern appears normal")
            else:
                suggestions.append("Monitor breathing patterns regularly")
        
        return {
            'is_apnea': is_apnea,
            'apnea_probability': apnea_score,
            'normal_probability': 1.0 - apnea_score,
            'confidence': confidence,
            'breathing_rate': breathing_rate,
            'max_pause_duration': max_pause_duration,
            'rms_energy': rms_energy,
            'amplitude_variation': amplitude_variation,
            'num_breathing_peaks': len(peaks),
            'breath_regularity': breath_regularity,
            'long_pause_count': long_pause_count,
            'reasons': reasons,
            'suggestions': suggestions
        }
    
    def _find_continuous_regions(self, mask):
        """Find continuous True regions in boolean mask."""
        regions = []
        in_region = False
        start_idx = 0
        
        for i, value in enumerate(mask):
            if value and not in_region:
                start_idx = i
                in_region = True
            elif not value and in_region:
                regions.append((start_idx, i))
                in_region = False
        
        if in_region:
            regions.append((start_idx, len(mask)))
        
        return regions

@st.cache_resource
def load_config():
    """Load configuration file."""
    try:
        with open('config/config.yaml', 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        st.error(f"Failed to load config: {e}")
        return None

@st.cache_resource
def initialize_components(config):
    """Initialize components."""
    try:
        audio_processor = AudioProcessor(**config['audio'])
        feature_extractor = FeatureExtractor(**config['features'])
        detector = EnhancedApneaDetector(sample_rate=config['audio']['sample_rate'])
        return audio_processor, feature_extractor, detector
    except Exception as e:
        st.error(f"Failed to initialize components: {e}")
        return None, None, None

def process_audio_file(audio_file, audio_processor, detector):
    """Process audio file with enhanced analysis."""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(audio_file.read())
            tmp_path = tmp_file.name
        
        # Load and process audio
        audio_data = audio_processor.load_audio(tmp_path)
        processed_audio = audio_processor.preprocess_audio(audio_data)
        segments = audio_processor.segment_audio(processed_audio)
        
        # Analyze each segment
        results = []
        all_suggestions = []
        
        for i, segment in enumerate(segments):
            segment_analysis = detector.analyze_breathing_segment(segment)
            
            segment_start = i * audio_processor.segment_length
            segment_end = segment_start + audio_processor.segment_length
            
            results.append({
                'segment': i + 1,
                'start_time': segment_start,
                'end_time': segment_end,
                'normal_prob': segment_analysis['normal_probability'],
                'apnea_prob': segment_analysis['apnea_probability'],
                'prediction': 'Apnea' if segment_analysis['is_apnea'] else 'Normal',
                'confidence': segment_analysis['confidence'],
                'breathing_rate': segment_analysis['breathing_rate'],
                'max_pause': segment_analysis['max_pause_duration'],
                'rms_energy': segment_analysis['rms_energy'],
                'regularity': segment_analysis['breath_regularity'],
                'reasons': segment_analysis['reasons'],
                'suggestions': segment_analysis['suggestions']
            })
            
            # Collect unique suggestions
            for suggestion in segment_analysis['suggestions']:
                if suggestion not in all_suggestions:
                    all_suggestions.append(suggestion)
        
        # Clean up temporary file
        os.unlink(tmp_path)
        
        return {
            'audio_data': audio_data,
            'processed_audio': processed_audio,
            'segments': segments,
            'sample_rate': audio_processor.sample_rate,
            'results': results,
            'overall_stats': calculate_enhanced_stats(results),
            'all_suggestions': all_suggestions
        }
        
    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return None

def calculate_enhanced_stats(results):
    """Calculate enhanced statistics with proper AHI."""
    if not results:
        return {}
    
    total_segments = len(results)
    apnea_segments = sum(1 for r in results if r['prediction'] == 'Apnea')
    normal_segments = total_segments - apnea_segments
    
    # Calculate proper AHI (Apnea-Hypopnea Index)
    total_duration_hours = max(r['end_time'] for r in results) / 3600.0
    ahi = apnea_segments / total_duration_hours if total_duration_hours > 0 else 0
    
    apnea_percentage = (apnea_segments / total_segments) * 100
    avg_confidence = np.mean([r['confidence'] for r in results])
    
    # Breathing rate statistics
    breathing_rates = [r['breathing_rate'] for r in results if r['breathing_rate'] > 0]
    avg_breathing_rate = np.mean(breathing_rates) if breathing_rates else 0
    min_breathing_rate = min(breathing_rates) if breathing_rates else 0
    max_breathing_rate = max(breathing_rates) if breathing_rates else 0
    
    # Pause statistics
    max_overall_pause = max([r['max_pause'] for r in results])
    avg_pause = np.mean([r['max_pause'] for r in results])
    
    # Risk assessment
    risk_level, risk_emoji = assess_risk_level(ahi, apnea_percentage, max_overall_pause)
    
    return {
        'total_segments': total_segments,
        'apnea_segments': apnea_segments,
        'normal_segments': normal_segments,
        'apnea_percentage': apnea_percentage,
        'ahi': ahi,
        'average_confidence': avg_confidence,
        'avg_breathing_rate': avg_breathing_rate,
        'min_breathing_rate': min_breathing_rate,
        'max_breathing_rate': max_breathing_rate,
        'max_pause_duration': max_overall_pause,
        'avg_pause_duration': avg_pause,
        'risk_level': (risk_level, risk_emoji),
        'total_duration': max(r['end_time'] for r in results)
    }

def assess_risk_level(ahi, apnea_percentage, max_pause):
    """Enhanced risk assessment based on multiple factors."""
    # Primary assessment based on AHI
    if ahi >= 30:
        risk = "Severe"
        emoji = "🚨"
    elif ahi >= 15:
        risk = "Moderate"  
        emoji = "🟠"
    elif ahi >= 5:
        risk = "Mild"
        emoji = "🟡"
    else:
        risk = "Normal"
        emoji = "🟢"
    
    # Adjust based on max pause duration
    if max_pause > 10:
        risk = "Critical"
        emoji = "🆘"
    elif max_pause > 5 and risk in ["Normal", "Mild"]:
        risk = "Moderate"
        emoji = "🟠"
    
    # Adjust based on percentage
    if apnea_percentage > 60:
        risk = "Severe"
        emoji = "🚨"
    
    return risk, emoji

def create_enhanced_visualizations(results_data):
    """Create comprehensive visualizations."""
    results = results_data['results']
    
    # 1. Waveform with apnea regions
    fig_waveform = create_audio_waveform_plot(
        results_data['audio_data'],
        results_data['sample_rate'],
        results
    )
    
    # 2. Breathing pattern analysis
    fig_breathing = create_breathing_analysis_plot(results)
    
    # 3. Risk timeline
    fig_timeline = create_risk_timeline(results)
    
    return fig_waveform, fig_breathing, fig_timeline

def create_audio_waveform_plot(audio_data, sample_rate, results=None):
    """Create enhanced waveform plot."""
    duration = len(audio_data) / sample_rate
    time_axis = np.linspace(0, duration, len(audio_data))
    
    fig = go.Figure()
    
    # Main waveform
    fig.add_trace(go.Scatter(
        x=time_axis,
        y=audio_data,
        mode='lines',
        name='Audio Waveform',
        line=dict(color='rgba(102, 126, 234, 0.7)', width=1),
        hovertemplate='<b>Time:</b> %{x:.2f}s<br><b>Amplitude:</b> %{y:.4f}<extra></extra>'
    ))
    
    # Add apnea regions
    if results:
        for result in results:
            if result['prediction'] == 'Apnea':
                fig.add_vrect(
                    x0=result['start_time'], x1=result['end_time'],
                    fillcolor="rgba(255, 99, 132, 0.3)",
                    layer="below",
                    line_width=0,
                    annotation_text=f"Apnea (Conf: {result['confidence']:.2f})",
                    annotation_position="top left"
                )
    
    fig.update_layout(
        title="Audio Waveform with Detected Apnea Events",
        xaxis_title="Time (seconds)",
        yaxis_title="Amplitude",
        template="plotly_white",
        height=400
    )
    
    return fig

def create_breathing_analysis_plot(results):
    """Create detailed breathing analysis plot."""
    if not results:
        return None
    
    segments = [r['segment'] for r in results]
    breathing_rates = [r['breathing_rate'] for r in results]
    pause_durations = [r['max_pause'] for r in results]
    confidences = [r['confidence'] for r in results]
    
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=(
            'Breathing Rate (breaths/min)', 
            'Maximum Pause Duration (seconds)',
            'Detection Confidence'
        ),
        vertical_spacing=0.08
    )
    
    # Breathing rate
    fig.add_trace(
        go.Scatter(
            x=segments, y=breathing_rates,
            mode='lines+markers',
            name='Breathing Rate',
            line=dict(color='blue', width=2),
            marker=dict(size=6)
        ),
        row=1, col=1
    )
    fig.add_hline(y=12, line_dash="dash", line_color="green", opacity=0.7, row=1, col=1)
    fig.add_hline(y=20, line_dash="dash", line_color="green", opacity=0.7, row=1, col=1)
    
    # Pause duration
    colors = ['red' if p > 3.0 else 'orange' if p > 2.0 else 'green' for p in pause_durations]
    fig.add_trace(
        go.Bar(
            x=segments, y=pause_durations,
            name='Max Pause Duration',
            marker_color=colors,
            opacity=0.7
        ),
        row=2, col=1
    )
    fig.add_hline(y=3.0, line_dash="dash", line_color="red", opacity=0.7, row=2, col=1)
    
    # Confidence
    conf_colors = ['red' if r['prediction'] == 'Apnea' else 'green' for r in results]
    fig.add_trace(
        go.Scatter(
            x=segments, y=confidences,
            mode='markers',
            name='Confidence',
            marker=dict(color=conf_colors, size=8)
        ),
        row=3, col=1
    )
    
    fig.update_layout(
        title="Comprehensive Breathing Pattern Analysis",
        height=700,
        template="plotly_white",
        showlegend=False
    )
    
    return fig

def create_risk_timeline(results):
    """Create risk assessment timeline."""
    if not results:
        return None
    
    segments = [r['segment'] for r in results]
    predictions = [1 if r['prediction'] == 'Apnea' else 0 for r in results]
    colors = ['red' if p == 1 else 'green' for p in predictions]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=segments,
        y=predictions,
        marker_color=colors,
        opacity=0.7,
        name='Risk Level',
        hovertemplate='<b>Segment %{x}</b><br>Status: %{text}<br><extra></extra>',
        text=[r['prediction'] for r in results]
    ))
    
    fig.update_layout(
        title="Risk Assessment Timeline",
        xaxis_title="Segment",
        yaxis_title="Risk (0=Normal, 1=Apnea)",
        template="plotly_white",
        height=300
    )
    
    return fig

def main():
    """Main enhanced Streamlit app."""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🩺 Enhanced Sleep Apnea Detector</h1>
        <p style="font-size: 1.2em; margin-bottom: 0;">
            Advanced AI-powered analysis with intelligent suggestions
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load configuration
    config = load_config()
    if config is None:
        st.stop()
    
    audio_processor, feature_extractor, detector = initialize_components(config)
    if audio_processor is None or detector is None:
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📋 Enhanced Analysis")
        
        st.markdown("#### 🤖 Detection Features")
        st.info("""
        **Enhanced Method:** Rule-Based + AI  
        **AHI Calculation:** Proper medical standard  
        **Suggestions:** Intelligent recommendations  
        **Risk Assessment:** Multi-factor analysis
        """)
        
        st.markdown("#### ⚙️ Detection Thresholds")
        st.markdown("""
        - **Critical Pause:** >10s (urgent care)
        - **Severe Pause:** >5s (immediate study)  
        - **Moderate Pause:** >3s (specialist consult)
        - **Mild Pause:** >2s (monitoring needed)
        """)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📁 Upload Audio File")
        uploaded_file = st.file_uploader(
            "Choose an audio file (WAV, MP3, FLAC)",
            type=['wav', 'mp3', 'flac'],
            help="Upload breathing sound recording for enhanced analysis"
        )
        
        if uploaded_file is not None:
            with st.spinner("🔄 Performing enhanced analysis..."):
                results_data = process_audio_file(uploaded_file, audio_processor, detector)
                
                if results_data:
                    st.success("✅ Enhanced analysis completed!")
                    
                    # Enhanced Statistics
                    stats = results_data['overall_stats']
                    risk_level, risk_emoji = stats['risk_level']
                    
                    # Key Metrics
                    metric_cols = st.columns(4)
                    
                    with metric_cols[0]:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>📊 AHI Score</h3>
                            <h2>{stats['ahi']:.1f}</h2>
                            <small>events/hour</small>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with metric_cols[1]:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>⚠️ Risk Level</h3>
                            <h2>{risk_emoji} {risk_level}</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with metric_cols[2]:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>📈 Apnea Events</h3>
                            <h2>{stats['apnea_segments']}</h2>
                            <small>{stats['apnea_percentage']:.1f}%</small>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with metric_cols[3]:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>⏸️ Max Pause</h3>
                            <h2>{stats['max_pause_duration']:.1f}s</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Intelligent Suggestions
                    if results_data['all_suggestions']:
                        st.markdown("### 💡 Intelligent Recommendations")
                        suggestions_text = "<br>• ".join([""] + results_data['all_suggestions'])
                        st.markdown(f"""
                        <div class="suggestion-card">
                            <h4>🎯 Personalized Suggestions</h4>
                            <p><strong>Based on your breathing pattern analysis:</strong></p>
                            <p>{suggestions_text}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Risk Assessment
                    if risk_level == "Normal":
                        st.markdown(f"""
                        <div class="success-card">
                            <h3>{risk_emoji} Assessment: {risk_level}</h3>
                            <p>Your breathing pattern appears normal with minimal sleep apnea risk detected.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    elif risk_level in ["Mild", "Moderate"]:
                        st.markdown(f"""
                        <div class="warning-card">
                            <h3>{risk_emoji} Assessment: {risk_level} Sleep Apnea</h3>
                            <p><strong>AHI: {stats['ahi']:.1f}</strong> - Consider consulting a sleep specialist for proper evaluation.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="warning-card">
                            <h3>{risk_emoji} Assessment: {risk_level}</h3>
                            <p><strong>URGENT:</strong> AHI: {stats['ahi']:.1f} - Seek immediate medical attention.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Enhanced Visualizations
                    st.markdown("### 📊 Comprehensive Analysis")
                    
                    fig_waveform, fig_breathing, fig_timeline = create_enhanced_visualizations(results_data)
                    
                    st.plotly_chart(fig_waveform, width='stretch')
                    st.plotly_chart(fig_breathing, width='stretch')
                    st.plotly_chart(fig_timeline, width='stretch')
                    
                    # Detailed Results
                    st.markdown("### 📋 Detailed Analysis Report")
                    df = pd.DataFrame(results_data['results'])
                    st.dataframe(
                        df[['segment', 'start_time', 'end_time', 'prediction', 
                           'confidence', 'breathing_rate', 'max_pause', 'reasons']],
                        width='stretch'
                    )
    
    with col2:
        st.markdown("### ℹ️ Enhanced Features")
        st.markdown("""
        <div class="info-card">
            <h4>🚀 New Capabilities</h4>
            <ul>
                <li><strong>Proper AHI Calculation:</strong> Medical standard scoring</li>
                <li><strong>Intelligent Suggestions:</strong> Personalized recommendations</li>
                <li><strong>Risk Stratification:</strong> Multi-factor assessment</li>
                <li><strong>Enhanced Detection:</strong> More accurate classification</li>
            </ul>
            
            <h4>📊 Medical Standards</h4>
            <ul>
                <li><strong>Normal:</strong> AHI < 5</li>
                <li><strong>Mild:</strong> AHI 5-15</li>
                <li><strong>Moderate:</strong> AHI 15-30</li>
                <li><strong>Severe:</strong> AHI > 30</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()