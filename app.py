# app.py
# Interactive Heartbeat Stethoscope Simulator with AI Heart Murmur Detection
#
# Features
# - Upload .wav or .mp3 heartbeat audio
# - Generate mel spectrogram
# - Segment the audio and compute a simple "abnormality score" per segment (heuristic)
# - AI-powered heart murmur detection using Dual Bayesian ResNet
# - Interactive spectrogram (hover for dB values) with highlighted abnormal segments
# - Inspect segment table and play any segment's audio
#
# How it works:
# 1. Heuristic analysis: Simple features (spectral flux, ZCR, envelope regularity)
# 2. AI analysis: Dual Bayesian ResNet for heart murmur classification
#
# To run:
#   pip install -r requirements.txt
#   (Install ffmpeg for mp3 support)
#   streamlit run app.py
#
# DISCLAIMER: Educational demo only. NOT medical advice or a diagnostic tool.

import io
import numpy as np
import streamlit as st
import librosa
import plotly.graph_objects as go
import plotly.express as px
import soundfile as sf
from typing import Tuple, List
import pandas as pd

# Import our heart murmur detector
from heart_murmur_detector import HeartMurmurDetector

st.set_page_config(
    page_title="stethicope type shi", 
    layout="wide",
    initial_sidebar_state="expanded"  # Better for both PC and mobile
)

# Universal responsive CSS (PC + Mobile)
st.markdown("""
<style>
    /* Universal improvements for all devices */
    .stButton > button {
        min-height: 44px;
        min-width: 44px;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    /* Responsive sidebar for all devices */
    .css-1d391kg {
        min-width: 250px;
    }
    
    /* Responsive design that works on both PC and mobile */
    @media (max-width: 768px) {
        /* Mobile-specific optimizations */
        .main .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
        
        .stButton > button {
            width: 100%;
            height: 3rem;
            font-size: 1.1rem;
            margin: 0.5rem 0;
        }
        
        .stRadio > div {
            flex-direction: column;
        }
        
        .stSlider > div {
            margin: 1rem 0;
        }
        
        .stTabs > div > div > div > div {
            font-size: 1rem;
            padding: 0.5rem;
        }
        
        /* Mobile spectrograms */
        .js-plotly-plot {
            height: 300px !important;
        }
    }
    
    @media (min-width: 769px) {
        /* Desktop-specific enhancements */
        .main .block-container {
            padding-left: 2rem;
            padding-right: 2rem;
        }
        
        .stButton > button {
            padding: 0.5rem 1rem;
            font-size: 1rem;
        }
        
        /* Desktop spectrograms */
        .js-plotly-plot {
            height: 500px !important;
        }
    }
    
    /* Universal theme consistency */
    .stRadio > div > div {
        background: transparent;
    }
    
    .stSlider > div > div > div > div {
        background: #1f77b4;
    }
    
    .stTabs > div > div > div > div[data-baseweb="tab"] {
        background: transparent;
        color: inherit;
    }
</style>
""", unsafe_allow_html=True)



st.title("🩺 yes this is an online stethicope cuz APPARANTLY my projects were a waste of time")
st.caption("i made this to show my worth, go on and try beat me")

# Initialize heart murmur detector
@st.cache_resource
def get_murmur_detector():
    # Use the trained models from the training script
    return HeartMurmurDetector(model_path="trained_models")

murmur_detector = get_murmur_detector()

# Sidebar
with st.sidebar:
    st.header("Audio Input")
    
    # Audio input method selection
    input_method = st.radio(
        "Choose input method:",
        ["Upload File", "Record with Microphone"]
    )
    
    if input_method == "Upload File":
        uploaded = st.file_uploader("Heartbeat audio (.wav or .mp3)", type=["wav", "mp3"])
        st.markdown("**Tips**")
        st.write("- Use a short clip (5–20 seconds)")
        st.write("- Place mic close to chest for clarity")
        recorded_audio = None
        
    else:  # Record with Microphone
        uploaded = None
        st.markdown("**Microphone Recording**")
        st.write("- Click 'Start Recording' to begin")
        st.write("- Speak/record for 5-20 seconds")
        st.write("- Click 'Stop Recording' when done")
        st.write("- Place mic close to chest for heart sounds")
        
        # Microphone recording instructions
        st.info("🎤 **To record with microphone:**\n"
                "1. Use your system's audio recording app (Windows Voice Recorder, etc.)\n"
                "2. Record heart sounds for 5-20 seconds\n"
                "3. Save as .wav file\n"
                "4. Upload the recorded file below\n\n"
                "**Alternative:** Use the 'Upload File' option with any heart sound recording")
        
        # Chrome built-in microphone recording
        st.markdown("**🎤 Chrome Microphone Recording**")
        
        # HTML5 Audio Recorder Component (Universal - PC + Mobile)
        audio_recorder_html = """
        <div class="audio-recorder-universal" style="text-align: center; padding: clamp(20px, 3vw, 25px); border: 2px solid #1f77b4; border-radius: 15px; background: linear-gradient(135deg, #2c3e50, #34495e); color: white; box-shadow: 0 8px 32px rgba(0,0,0,0.3);">
            <h3 style="color: #ecf0f1; margin-bottom: 25px; font-size: clamp(20px, 3vw, 24px); text-shadow: 0 2px 4px rgba(0,0,0,0.3);">🎤 Browser Microphone Recorder</h3>
            <div id="controls">
                <button id="recordButton" style="background: linear-gradient(135deg, #e74c3c, #c0392b); color: white; padding: clamp(12px, 2vw, 18px) clamp(20px, 3vw, 35px); border: none; border-radius: 30px; font-size: clamp(14px, 2.5vw, 18px); font-weight: bold; cursor: pointer; margin: 10px; box-shadow: 0 4px 15px rgba(231, 76, 60, 0.4); transition: all 0.3s ease; text-shadow: 0 1px 2px rgba(0,0,0,0.3); min-height: 44px; min-width: 44px;">
                    🎤 Start Recording
                </button>
                <button id="stopButton" style="background: linear-gradient(135deg, #95a5a6, #7f8c8d); color: white; padding: clamp(12px, 2vw, 18px) clamp(20px, 3vw, 35px); border: none; border-radius: 30px; font-size: clamp(14px, 2.5vw, 18px); font-weight: bold; cursor: pointer; margin: 10px; box-shadow: 0 4px 15px rgba(149, 165, 166, 0.4); transition: all 0.3s ease; text-shadow: 0 1px 2px rgba(0,0,0,0.3); display: none; min-height: 44px; min-width: 44px;">
                    ⏹️ Stop Recording
                </button>
            </div>
            <div id="status" style="margin: 20px; font-size: clamp(14px, 2.5vw, 16px); color: #bdc3c7; font-weight: 500;"></div>
            <audio id="audioPlayback" controls style="width: 100%; max-width: 400px; margin: 20px; border-radius: 10px; background: #34495e;"></audio>
            <div id="downloadSection" style="margin: 20px; display: none;">
                <button id="downloadButton" style="background: linear-gradient(135deg, #3498db, #2980b9); color: white; padding: clamp(8px, 1.5vw, 12px) clamp(15px, 2.5vw, 25px); border: none; border-radius: 20px; cursor: pointer; font-weight: bold; box-shadow: 0 4px 15px rgba(52, 152, 219, 0.4); transition: all 0.3s ease; text-shadow: 0 1px 2px rgba(0,0,0,0.3); min-height: 44px;">
                    💾 Download Recording
                </button>
            </div>
        </div>
        
        <script>
        let mediaRecorder;
        let audioChunks = [];
        let audioBlob;
        
        // Add hover effects
        document.addEventListener('DOMContentLoaded', function() {
            const recordBtn = document.getElementById('recordButton');
            const stopBtn = document.getElementById('stopButton');
            const downloadBtn = document.getElementById('downloadButton');
            
            // Hover effects for buttons
            [recordBtn, stopBtn, downloadBtn].forEach(btn => {
                if (btn) {
                    btn.addEventListener('mouseenter', function() {
                        this.style.transform = 'translateY(-2px)';
                        this.style.boxShadow = '0 6px 20px rgba(0,0,0,0.4)';
                    });
                    
                    btn.addEventListener('mouseleave', function() {
                        this.style.transform = 'translateY(0)';
                        this.style.boxShadow = this.id === 'recordButton' ? '0 4px 15px rgba(231, 76, 60, 0.4)' : 
                                              this.id === 'stopButton' ? '0 4px 15px rgba(149, 165, 166, 0.4)' : 
                                              '0 4px 15px rgba(52, 152, 219, 0.4)';
                    });
                }
            });
        });
        
        document.getElementById('recordButton').addEventListener('click', startRecording);
        document.getElementById('stopButton').addEventListener('click', stopRecording);
        document.getElementById('downloadButton').addEventListener('click', downloadRecording);
        
        async function startRecording() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new MediaRecorder(stream);
                audioChunks = [];
                
                mediaRecorder.ondataavailable = (event) => {
                    audioChunks.push(event.data);
                };
                
                mediaRecorder.onstop = () => {
                    audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                    const audioUrl = URL.createObjectURL(audioBlob);
                    document.getElementById('audioPlayback').src = audioUrl;
                    document.getElementById('downloadSection').style.display = 'block';
                    document.getElementById('status').innerHTML = '✅ Recording completed! Listen and download below.';
                };
                
                mediaRecorder.start();
                document.getElementById('recordButton').style.display = 'none';
                document.getElementById('stopButton').style.display = 'inline-block';
                document.getElementById('status').innerHTML = '🎤 Recording... Speak into your microphone!';
                
            } catch (err) {
                document.getElementById('status').innerHTML = '❌ Error: ' + err.message + '<br>Please allow microphone access.';
            }
        }
        
        function stopRecording() {
            if (mediaRecorder && mediaRecorder.state !== 'inactive') {
                mediaRecorder.stop();
                mediaRecorder.stream.getTracks().forEach(track => track.stop());
                
                document.getElementById('recordButton').style.display = 'inline-block';
                document.getElementById('stopButton').style.display = 'none';
            }
        }
        
        function downloadRecording() {
            if (audioBlob) {
                const url = URL.createObjectURL(audioBlob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'heart_sound_recording.wav';
                a.click();
                URL.revokeObjectURL(url);
            }
        }
        </script>
        """
        
        # Display the HTML recorder
        st.components.v1.html(audio_recorder_html, height=400)
        
        st.markdown("---")
        st.markdown("**📁 Or upload a pre-recorded file:**")
        
        # File uploader for recorded audio
        recorded_audio = st.file_uploader(
            "Upload recorded audio (.wav)", 
            type=["wav"], 
            key="recorded_audio_uploader",
            help="Upload your microphone recording here"
        )
        
        if recorded_audio is not None:
            st.success("✅ Recorded audio uploaded successfully!")
            st.audio(recorded_audio, format="audio/wav")
    
    st.divider()
    
    st.header("Analysis Type")
    analysis_type = st.radio(
        "Choose analysis method:",
        ["Heuristic Analysis", "AI Heart Murmur Detection", "Both"]
    )
    
    if analysis_type in ["Heuristic Analysis", "Both"]:
        st.subheader("Heuristic Settings")
        seg_dur = st.slider("Segment duration (seconds)", 0.3, 2.0, 1.0, 0.1)
        seg_hop = st.slider("Segment hop (seconds)", 0.1, 1.0, 0.5, 0.1)
        mel_bins = st.slider("Mel bins", 32, 256, 128, 8)
        fmax = st.slider("Max frequency (Hz)", 500, 4000, 2000, 100)
        top_percent = st.slider("Highlight top-% most abnormal", 5, 60, 20, 5)
        score_threshold = st.slider("Absolute score threshold", 0.0, 1.0, 0.60, 0.01)
    
    if analysis_type in ["AI Heart Murmur Detection", "Both"]:
        st.subheader("AI Settings")
        mc_samples = st.slider("Monte Carlo samples", 5, 50, 20, 5)
        show_uncertainty = st.checkbox("Show uncertainty analysis", value=True)

def load_audio(file) -> Tuple[np.ndarray, int]:
    # librosa.load uses audioread backend, which can decode mp3 if ffmpeg is available.
    y, sr = librosa.load(file, sr=None, mono=True)
    # Normalize
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))
    return y, sr

def load_audio_from_bytes(audio_bytes) -> Tuple[np.ndarray, int]:
    """Load audio from recorded bytes (from microphone)"""
    # Convert bytes to numpy array
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True)
    # Normalize
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))
    return y, sr

def compute_mel_spectrogram(y: np.ndarray, sr: int, n_mels: int, fmax: int):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=fmax)
    S_db = librosa.power_to_db(S, ref=np.max)
    times = librosa.times_like(S_db, sr=sr, hop_length=512)  # default hop_length for melspectrogram
    freqs = librosa.mel_frequencies(n_mels=n_mels, fmin=0, fmax=fmax)
    return S_db, times, freqs

def segment_indices(sr: int, n_samples: int, seg_dur: float, hop_dur: float):
    seg_len = int(seg_dur * sr)
    hop_len = int(hop_dur * sr)
    starts = list(range(0, max(1, n_samples - seg_len + 1), max(1, hop_len)))
    if len(starts) == 0:
        starts = [0]
    return [(s, min(s + seg_len, n_samples)) for s in starts]

def envelope_and_onsets(y: np.ndarray, sr: int):
    # Onset strength is a good generic envelope proxy
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    times = librosa.times_like(onset_env, sr=sr)
    return onset_env, times

def segment_scores(y: np.ndarray, sr: int, segs: List[Tuple[int,int]]):
    scores = []
    for (a, b) in segs:
        seg = y[a:b]
        # Features
        # (1) Spectral flux: mean positive change in spectrum
        S = np.abs(librosa.stft(seg, n_fft=1024, hop_length=256))
        flux = np.diff(S, axis=1)
        flux = np.maximum(flux, 0).mean() if flux.size else 0.0

        # (2) Zero crossing rate (ZCR): noisier / breathy / murmur sections may have higher zcr
        zcr = librosa.feature.zero_crossing_rate(seg, frame_length=1024, hop_length=256).mean()

        # (3) Envelope regularity: estimate inter-onset intervals variability (irregular = more abnormal)
        onset_env = librosa.onset.onset_strength(y=seg, sr=sr)
        if len(onset_env) > 4:
            # Peak picking
            peaks = librosa.util.peak_pick(onset_env, pre_max=2, post_max=2, pre_avg=2, post_avg=2, delta=0.2, wait=2)
            if len(peaks) >= 3:
                intervals = np.diff(peaks)
                if np.mean(intervals) > 0:
                    irr = np.std(intervals) / (np.mean(intervals) + 1e-6)
                else:
                    irr = 1.0
            else:
                irr = 0.8  # uncertain → moderate irregularity
        else:
            irr = 0.5

        # Normalize features roughly
        # Scale factors chosen empirically for demo stability
        flux_n = np.tanh(flux / 5.0)
        zcr_n = np.tanh(zcr * 10.0)
        irr_n = np.tanh(irr)

        score = float(0.45*irr_n + 0.35*flux_n + 0.20*zcr_n)
        scores.append(score)
    return np.array(scores, dtype=float)

def build_spectrogram_figure(S_db, times, freqs, segs, seg_scores, top_pct, abs_thresh):
    fig = go.Figure(data=go.Heatmap(
        z=S_db,
        x=times,
        y=freqs,
        colorscale='Viridis',
        colorbar=dict(title='dB')
    ))
    fig.update_layout(
        xaxis_title="Time (s)",
        yaxis_title="Mel Frequency (Hz)",
        title="Mel Spectrogram",
        height=500,
    )

    # Determine which segments to highlight
    if len(seg_scores):
        n_top = max(1, int(len(seg_scores) * top_pct / 100.0))
        # Indices of top-N scores
        top_idx = np.argsort(seg_scores)[-n_top:]
        # Also mark any above absolute threshold
        abs_idx = np.where(seg_scores >= abs_thresh)[0]
        mark = set(top_idx.tolist()) | set(abs_idx.tolist())
    else:
        mark = set()

    for i, (a, b) in enumerate(segs):
        start_t = a_time = a / sr_global
        end_t = b_time = b / sr_global
        if i in mark:
            fig.add_vrect(x0=start_t, x1=end_t, fillcolor="red", opacity=0.25, line_width=0)

    return fig

def slice_to_wav_bytes(y_slice: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, y_slice, sr, format='WAV')
    buf.seek(0)
    return buf.read()

st.markdown("Upload a heartbeat recording or record with your microphone to analyze. Choose your analysis method below.")

# Microphone recording tips
if input_method == "Record with Microphone":
    st.info("🎤 **Chrome Microphone Recording Tips:**\n"
            "• Allow microphone access when prompted\n"
            "• Place microphone close to chest (over heart area)\n"
            "• Record for 5-20 seconds for best results\n"
            "• Stay quiet and minimize background noise\n"
            "• Use the browser recorder below or upload a file\n\n"
            "📱 **Mobile Users:** Touch-friendly buttons optimized for your device!", icon="🎤")

# Check if we have audio input (either uploaded or recorded)
audio_available = uploaded is not None or recorded_audio is not None

if not audio_available:
    if input_method == "Record with Microphone":
        st.info("🎤 Click 'Start Recording' above to begin recording with your microphone.", icon="🎤")
    else:
        st.info("📁 Upload a .wav or .mp3 file to begin.", icon="📁")
else:
    with st.spinner("Loading audio…"):
        try:
            if uploaded is not None:
                # Load uploaded file
                y, sr = load_audio(uploaded)
                st.success(f"✅ Loaded uploaded audio: {uploaded.name}")
            else:
                # Load recorded audio
                y, sr = load_audio_from_bytes(recorded_audio)
                st.success("✅ Loaded recorded audio from microphone")
        except Exception as e:
            st.error(f"Could not read audio: {e}")
            st.stop()

    # Keep sr globally for vrect time mapping
    global sr_global
    sr_global = sr

    # Create tabs for different analysis methods
    if analysis_type == "Heuristic Analysis":
        tab1, tab2 = st.tabs(["Heuristic Analysis", "Audio Playback"])
    elif analysis_type == "AI Heart Murmur Detection":
        tab1, tab2 = st.tabs(["AI Murmur Detection", "Audio Playback"])
    else:  # Both
        tab1, tab2, tab3 = st.tabs(["Heuristic Analysis", "AI Murmur Detection", "Audio Playback"])

    # Tab 1: Heuristic Analysis
    if analysis_type in ["Heuristic Analysis", "Both"]:
        with tab1:
            st.subheader("🔍 Heuristic Analysis")
            st.caption("Simple feature-based abnormality detection using spectral analysis")
            
            # Spectrogram
            with st.spinner("Computing spectrogram…"):
                S_db, times, freqs = compute_mel_spectrogram(y, sr, n_mels=mel_bins, fmax=fmax)

            # Segmentation
            segs = segment_indices(sr, len(y), seg_dur, seg_hop)

            # Scores
            with st.spinner("Analyzing segments…"):
                scores = segment_scores(y, sr, segs)

            # Build interactive spectrogram
            fig = build_spectrogram_figure(S_db, times, freqs, segs, scores, top_percent, score_threshold)
            st.plotly_chart(fig, use_container_width=True)

            # Segment table
            rows = []
            for i, (a, b) in enumerate(segs):
                rows.append({
                    "segment": i,
                    "start_s": round(a/sr, 3),
                    "end_s": round(b/sr, 3),
                    "duration_s": round((b-a)/sr, 3),
                    "score": round(float(scores[i]), 4)
                })
            df = pd.DataFrame(rows)
            # Flag highlighted segments
            n_top = max(1, int(len(scores) * top_percent / 100.0)) if len(scores) else 0
            top_idx = np.argsort(scores)[-n_top:] if n_top else []
            abs_idx = np.where(scores >= score_threshold)[0] if len(scores) else []
            highlighted = set(top_idx.tolist()) | set(abs_idx.tolist())
            df["highlighted"] = df["segment"].isin(highlighted)

            st.subheader("Segment Scores")
            st.caption("Higher score = more likely to be abnormal (heuristic). Use with caution.")
            st.dataframe(df, use_container_width=True)

            # Download results
            st.subheader("Export")
            st.download_button(
                label="Download segment scores (CSV)",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="heuristic_segment_scores.csv",
                mime="text/csv"
            )

    # Tab 2: AI Heart Murmur Detection
    if analysis_type in ["AI Heart Murmur Detection", "Both"]:
        with tab2 if analysis_type == "AI Heart Murmur Detection" else tab2:
            st.subheader("🤖 AI Heart Murmur Detection")
            st.caption("Dual Bayesian ResNet analysis with uncertainty estimation")
            
            # AI Analysis
            with st.spinner("Running AI analysis…"):
                try:
                    # Get detailed analysis
                    analysis_result = murmur_detector.get_detailed_analysis(y, sr)
                    
                    # Overall classification
                    overall = analysis_result['overall']
                    final_class = overall['final_classification']
                    
                    # Display results
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Heart Murmur Status", 
                            final_class['class'],
                            delta=f"{final_class['confidence']:.1%} confidence"
                        )
                    
                    with col2:
                        st.metric(
                            "Present Probability",
                            f"{final_class['present_probability']:.1%}",
                            delta="Likelihood of murmur"
                        )
                    
                    with col3:
                        st.metric(
                            "Overall Confidence",
                            f"{overall['overall_confidence']:.1%}",
                            delta="Model certainty"
                        )
                    
                    # Detailed model outputs
                    if show_uncertainty:
                        st.subheader("Model Uncertainty Analysis")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Present vs Other Model**")
                            model1 = overall['models']['present_vs_other']
                            st.write(f"Confidence: {model1['confidence']:.3f}")
                            st.write(f"Uncertainty: {np.mean(model1['uncertainty']):.3f}")
                        
                        with col2:
                            st.write("**Unknown vs Other Model**")
                            model2 = overall['models']['unknown_vs_other']
                            st.write(f"Confidence: {model2['confidence']:.3f}")
                            st.write(f"Uncertainty: {np.mean(model2['uncertainty']):.3f}")
                    
                    # Segment-level analysis
                    st.subheader("Segment-Level Analysis")
                    segment_data = []
                    
                    for segment_info, segment_result in analysis_result['segments']:
                        seg_class = segment_result['final_classification']
                        segment_data.append({
                            'start_time': f"{segment_info['start_time']:.2f}s",
                            'end_time': f"{segment_info['end_time']:.2f}s",
                            'classification': seg_class['class'],
                            'confidence': f"{seg_class['confidence']:.1%}",
                            'present_prob': f"{seg_class['present_probability']:.1%}",
                            'unknown_prob': f"{seg_class['unknown_probability']:.1%}"
                        })
                    
                    segment_df = pd.DataFrame(segment_data)
                    st.dataframe(segment_df, use_container_width=True)
                    
                    # Download AI results
                    st.subheader("Export AI Analysis")
                    st.download_button(
                        label="Download AI analysis results (CSV)",
                        data=segment_df.to_csv(index=False).encode("utf-8"),
                        file_name="ai_murmur_analysis.csv",
                        mime="text/csv"
                    )
                    
                except Exception as e:
                    st.error(f"AI analysis failed: {e}")
                    st.info("This might be due to model initialization. The AI models are randomly initialized for demo purposes.")
                    st.write("To use pre-trained models, you would need to:")
                    st.write("1. Train the models on heart sound data")
                    st.write("2. Save the model weights")
                    st.write("3. Update the model path in the HeartMurmurDetector")

    # Tab 3: Audio Playback (when both analyses are selected)
    if analysis_type == "Both":
        with tab3:
            st.subheader("🎵 Audio Playback")
            st.caption("Listen to specific segments of your recording")
            
            # Use heuristic segmentation for playback
            segs = segment_indices(sr, len(y), seg_dur, seg_hop)
            scores = segment_scores(y, sr, segs)
            
            # Playback controls
            sel = st.slider("Choose segment index", 0, max(0, len(segs)-1), 0, 1)
            a, b = segs[sel]
            y_slice = y[a:b]
            st.write(f"Selected: {sel}  |  {a/sr:.2f}s → {b/sr:.2f}s  |  Score: {scores[sel]:.3f}")
            st.audio(slice_to_wav_bytes(y_slice, sr), format="audio/wav")
    else:
        # Single tab for audio playback
        with tab2:
            st.subheader("🎵 Audio Playback")
            st.caption("Listen to specific segments of your recording")
            
            # Use default segmentation for playback when only AI analysis is selected
            if analysis_type == "AI Heart Murmur Detection":
                # Use default values for segmentation
                default_seg_dur = 1.0
                default_seg_hop = 0.5
                segs = segment_indices(sr, len(y), default_seg_dur, default_seg_hop)
                scores = segment_scores(y, sr, segs)
            else:
                # Use heuristic segmentation for playback
                segs = segment_indices(sr, len(y), seg_dur, seg_hop)
                scores = segment_scores(y, sr, segs)
            
            # Playback controls
            sel = st.slider("Choose segment index", 0, max(0, len(segs)-1), 0, 1)
            a, b = segs[sel]
            y_slice = y[a:b]
            st.write(f"Selected: {sel}  |  {a/sr:.2f}s → {b/sr:.2f}s  |  Score: {scores[sel]:.3f}")
            st.audio(slice_to_wav_bytes(y_slice, sr), format="audio/wav")

    # Save annotated spectrogram as static PNG for export
    try:
        import matplotlib.pyplot as plt
        import librosa.display  # for completeness, not used for plotting here
        # Use a static save via plotly
        st.caption("Tip: Use the Plotly toolbar (camera icon) to save an image of the spectrogram.")
    except Exception:
        pass

st.divider()
st.markdown("""
**Disclaimer:** i am tired, i m ! 55 ____ helep me
""")