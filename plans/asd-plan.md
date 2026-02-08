# 🎤 Active Speaker Detection Integration Project Plan

## 📋 Project Overview

**Objective**: Integrate Active Speaker Detection (ASD) to intelligently identify which face is speaking and apply AI mouth overlay only to the active speaker, eliminating face switching issues and improving semantic accuracy.

**Current Issues** (🔄 **TO BE RESOLVED**):
- Primary face locking prevents switching but doesn't know who is actually speaking
- Multi-speaker scenarios require manual face selection or spatial heuristics
- Face selection based on size/position rather than semantic understanding
- No audio-visual correlation for speaker identification
- Potential for applying lip-sync to non-speaking faces in group conversations

**Target Goals**:
- 🎯 **Semantic Speaker Detection** - Identify actual speaker using audio-visual correlation
- 🔄 **Dynamic Speaker Switching** - Handle multi-speaker scenarios intelligently
- 📊 **Confidence-Based Selection** - Combine ASD confidence with face detection confidence
- ⚡ **Real-time Performance** - Maintain current processing speeds
- 🎥 **Multi-Speaker Support** - Handle videos with multiple people speaking at different times

## 🏆 **Integration Benefits**

### **✅ Why ASD is the Right Solution**
- **Semantic Understanding**: Knows WHO is speaking, not just WHERE faces are
- **Multi-Speaker Handling**: Can switch between speakers when conversation changes
- **Audio-Visual Correlation**: Uses both audio energy and visual mouth movement
- **Confidence Weighting**: Provides probability scores for each detected face
- **Temporal Consistency**: Maintains speaker tracking across frames
- **Fallback Compatibility**: Works with existing face selection as backup

---

## 🔍 Phase 0: Face Detection Debug & Visualization Tool

**Duration**: 1-2 days  
**Risk Level**: Very Low  
**Goal**: Create comprehensive face detection visualization tool for troubleshooting and validation

### **Objectives**
- Build debug script to visualize YOLOv8 face detection pipeline
- Show ALL detected faces with bounding boxes and landmarks
- Create video output showing face tracking in real-time
- Provide foundation for ASD debugging and validation

### **Key Actions**
1. **Comprehensive Face Visualization**
   - Draw bounding boxes around ALL detected faces (not just selected one)
   - Display 5-point landmarks (eyes, nose, mouth corners) for each face
   - Show confidence scores for each detection
   - Use color coding to distinguish different faces

2. **Primary Face Selection Visualization**
   - Highlight which face is selected as "primary" for lip-sync
   - Show selection reasoning (confidence, size, center position, temporal consistency)
   - Display face locking status and primary face tracking
   - Visualize face switching events with clear indicators

3. **Frame-Level Statistics**
   - Display total number of faces detected per frame
   - Show frame number and timestamp
   - Add detection quality metrics
   - Track face consistency across frames

4. **Debug Output Generation**
   - Create full video with all visualizations overlaid
   - Support configurable visualization options
   - Add frame-by-frame analysis capabilities
   - Generate detection statistics and reports

### **Technical Implementation**
```python
class FaceDetectionDebugger:
    def __init__(self, config_path="configs/inference/test.yaml"):
        self.yolo_detector = YOLOv8_face()
        self.colors = self._generate_face_colors()
        self.frame_stats = []
        
    def create_debug_video(self, input_video, output_video, options=None):
        """Create comprehensive face detection debug video"""
        cap = cv2.VideoCapture(input_video)
        writer = self._setup_video_writer(output_video, cap)
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Run YOLOv8 detection on frame
            faces, confidences, landmarks = self.yolo_detector.get_detections_for_batch([frame])
            
            # Visualize ALL faces
            debug_frame = self._visualize_all_faces(frame, faces, confidences, landmarks, frame_idx)
            
            # Add frame statistics
            debug_frame = self._add_frame_statistics(debug_frame, faces, frame_idx)
            
            # Highlight primary face selection
            primary_face_idx = self._get_primary_face_selection(faces, confidences, landmarks)
            debug_frame = self._highlight_primary_face(debug_frame, faces, primary_face_idx)
            
            writer.write(debug_frame)
            frame_idx += 1
            
        self._cleanup_and_generate_report(cap, writer)
    
    def _visualize_all_faces(self, frame, faces, confidences, landmarks, frame_idx):
        """Draw bounding boxes, landmarks, and info for all detected faces"""
        debug_frame = frame.copy()
        
        for i, (face, conf, lm) in enumerate(zip(faces, confidences, landmarks)):
            color = self.colors[i % len(self.colors)]
            
            # Draw bounding box
            x1, y1, x2, y2 = face
            cv2.rectangle(debug_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            # Draw landmarks
            if lm is not None:
                for point in lm:
                    cv2.circle(debug_frame, (int(point[0]), int(point[1])), 3, color, -1)
            
            # Add face info
            info_text = f"Face {i}: {conf:.3f}"
            cv2.putText(debug_frame, info_text, (int(x1), int(y1)-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return debug_frame
    
    def _highlight_primary_face(self, frame, faces, primary_idx):
        """Highlight the selected primary face with special indicators"""
        if primary_idx is not None and primary_idx < len(faces):
            x1, y1, x2, y2 = faces[primary_idx]
            
            # Draw thick primary face border
            cv2.rectangle(frame, (int(x1)-5, int(y1)-5), (int(x2)+5, int(y2)+5), 
                         (0, 255, 0), 4)  # Green for primary
            
            # Add "PRIMARY" label
            cv2.putText(frame, "PRIMARY FACE", (int(x1), int(y1)-25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        return frame
    
    def _add_frame_statistics(self, frame, faces, frame_idx):
        """Add frame-level statistics overlay"""
        stats_text = [
            f"Frame: {frame_idx}",
            f"Faces Detected: {len(faces)}",
            f"Primary Face Locked: {'Yes' if self.yolo_detector.primary_face_locked else 'No'}"
        ]
        
        y_offset = 30
        for text in stats_text:
            cv2.putText(frame, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += 25
        
        return frame
```

### **Script Usage**
```bash
# Create face detection debug video
python debug_face_detection.py \
    --input_video "data/video/test_multi_speaker.mp4" \
    --output_video "debug_output/face_detection_analysis.mp4" \
    --show_landmarks true \
    --show_confidence true \
    --highlight_primary true \
    --color_faces true \
    --add_statistics true

# Configuration options
python debug_face_detection.py \
    --config "configs/inference/test.yaml" \
    --input_video "problematic_video.mp4" \
    --output_video "debug_face_switching.mp4" \
    --track_face_switches true \
    --generate_report true
```

### **Debug Features**
1. **Multi-Face Visualization**
   - Different colored bounding boxes for each detected face
   - Consistent face colors across frames for tracking
   - Face ID numbers for easy reference

2. **Landmark Accuracy Check**
   - 5-point landmarks displayed as colored dots
   - Mouth corner positioning validation
   - Eye and nose landmark verification

3. **Primary Face Selection Analysis**
   - Clear highlighting of selected face
   - Selection reasoning display
   - Face locking status indication

4. **Face Switching Detection**
   - Visual alerts when primary face changes
   - Switching reason display
   - Temporal consistency analysis

5. **Statistical Reporting**
   - Frame-by-frame face count
   - Confidence score distributions
   - Face switching frequency analysis
   - Detection quality metrics

### **Integration Benefits for ASD**
- **Baseline Validation**: Ensures YOLOv8 detection is working correctly before adding ASD
- **Multi-Face Analysis**: Shows all available faces that ASD will analyze
- **Debug Foundation**: Provides visualization framework for ASD confidence scores
- **Problem Identification**: Helps identify specific scenarios where face switching occurs
- **Quality Assurance**: Validates that face detection meets requirements for ASD input

### **Success Criteria**
- [x] Visualizes ALL detected faces with bounding boxes and landmarks
- [x] Clearly shows primary face selection and reasoning
- [x] Generates complete debug video with frame statistics
- [x] Provides configurable visualization options
- [x] Creates foundation for ASD debugging tools
- [x] Identifies face switching issues for ASD to solve

---

## 🚀 Phase 1: Lightweight Audio-Visual Correlation

**Duration**: 2-3 days  
**Risk Level**: Low  
**Goal**: Implement basic ASD using audio energy detection and visual motion correlation

### **Objectives**
- Detect audio activity (speech vs silence)
- Correlate audio energy with visual mouth movement
- Integrate with existing YOLOv8 face selection pipeline
- Maintain backward compatibility with current system

### **Key Actions**
1. **Audio Processing Integration**
   - Extract audio segments synchronized with video frames
   - Implement audio energy detection and voice activity detection (VAD)
   - Add audio preprocessing (noise reduction, normalization)
   - Create audio windowing system for frame-level analysis

2. **Visual Motion Detection**
   - Calculate mouth region movement between consecutive frames
   - Implement optical flow analysis for mouth area
   - Add visual activity scoring based on landmark changes
   - Create motion smoothing to reduce noise

3. **Audio-Visual Fusion**
   - Combine audio energy with visual motion scores
   - Implement weighted scoring system for speaker probability
   - Add temporal smoothing for consistent speaker detection
   - Create confidence thresholding for speaker selection

4. **Pipeline Integration**
   - Modify `YOLOv8_face._select_best_face()` to include ASD scoring
   - Add ASD parameters to YAML configuration
   - Implement fallback to existing face selection when ASD is uncertain
   - Add debug visualization for ASD scores

### **Technical Implementation**
```python
class LightweightASD:
    def __init__(self, audio_window_ms=500, motion_threshold=0.1):
        self.audio_window_ms = audio_window_ms
        self.motion_threshold = motion_threshold
        self.previous_landmarks = None
        
    def detect_active_speaker(self, faces, landmarks, audio_segment):
        # Audio energy analysis
        audio_energy = self._calculate_audio_energy(audio_segment)
        
        # Visual motion analysis
        motion_scores = self._calculate_motion_scores(faces, landmarks)
        
        # Audio-visual correlation
        av_scores = self._correlate_audio_visual(audio_energy, motion_scores)
        
        return np.argmax(av_scores) if max(av_scores) > self.confidence_threshold else None
```

### **Success Criteria**
- [ ] Audio energy detection with 95%+ accuracy for speech vs silence
- [ ] Visual motion correlation identifies mouth movement
- [ ] ASD integration maintains current processing speed
- [ ] Fallback to existing system when ASD is uncertain
- [ ] YAML configuration for all ASD parameters

---

## 🧠 Phase 2: Advanced TalkNet-Style ASD

**Duration**: 4-5 days  
**Risk Level**: Medium  
**Goal**: Implement sophisticated audio-visual feature fusion for robust speaker detection

### **Objectives**
- Deploy pre-trained TalkNet or similar ASD model
- Extract deep audio and visual features
- Implement neural network-based audio-visual fusion
- Add multi-speaker scenario handling

### **Key Actions**
1. **Model Integration**
   - Research and select optimal ASD model (TalkNet, SyncNet, or custom)
   - Implement ONNX conversion for inference optimization
   - Add model weight management and loading system
   - Create feature extraction pipelines for audio and visual data

2. **Feature Engineering**
   - Implement audio feature extraction (MFCC, spectrograms, embeddings)
   - Add visual feature extraction from face crops
   - Create temporal feature aggregation across multiple frames
   - Implement feature normalization and preprocessing

3. **Neural Network Fusion**
   - Deploy attention-based audio-visual fusion network
   - Add speaker embedding generation for identity consistency
   - Implement multi-head attention for robust feature correlation
   - Create confidence calibration for speaker predictions

4. **Multi-Speaker Handling**
   - Add speaker tracking across temporal sequences
   - Implement speaker change detection algorithms
   - Create smooth transitions between different speakers
   - Add support for overlapping speech scenarios

### **Technical Architecture**
```python
class AdvancedASD:
    def __init__(self, model_path="models/talknet_asd.onnx"):
        self.audio_encoder = AudioEncoder(model_path)
        self.visual_encoder = VisualEncoder(model_path)
        self.fusion_network = FusionNetwork(model_path)
        self.speaker_tracker = SpeakerTracker()
        
    def predict_speakers(self, faces, audio_segment, temporal_context=None):
        # Extract deep features
        audio_features = self.audio_encoder.extract(audio_segment)
        visual_features = [self.visual_encoder.extract(face) for face in faces]
        
        # Fusion and prediction
        speaker_probs = self.fusion_network.predict(audio_features, visual_features)
        
        # Temporal tracking
        tracked_speakers = self.speaker_tracker.update(speaker_probs, temporal_context)
        
        return tracked_speakers
```

### **Success Criteria**
- [ ] 90%+ accuracy on multi-speaker detection benchmarks
- [ ] Robust handling of speaker changes and overlapping speech
- [ ] Real-time inference performance (<100ms per frame)
- [ ] Smooth speaker transitions without jarring switches
- [ ] Integration with existing MuseTalk pipeline

---

## 🔧 Phase 3: Production Optimization & Integration

**Duration**: 2-3 days  
**Risk Level**: Low  
**Goal**: Optimize ASD system for production use and seamless MuseTalk integration

### **Objectives**
- Optimize inference performance for real-time processing
- Add comprehensive configuration and debugging tools
- Implement robust error handling and fallback mechanisms
- Create quality assessment and monitoring tools

### **Key Actions**
1. **Performance Optimization**
   - Implement batch processing for multiple faces
   - Add GPU acceleration for neural network inference
   - Optimize audio processing pipeline for minimal latency
   - Create efficient caching for repeated computations

2. **Configuration System**
   - Add comprehensive YAML configuration for all ASD parameters
   - Implement quality presets (fast/balanced/accurate)
   - Create runtime parameter adjustment capabilities
   - Add model selection options (lightweight vs advanced)

3. **Error Handling & Fallbacks**
   - Implement graceful degradation when ASD fails
   - Add automatic fallback to face-locking system
   - Create robust handling of audio/video sync issues
   - Implement model loading error recovery

4. **Monitoring & Debugging**
   - Add ASD confidence visualization overlays
   - Create speaker detection timeline visualization
   - Implement performance metrics and timing analysis
   - Add quality assessment tools for ASD accuracy

### **Configuration Example**
```yaml
# Active Speaker Detection Configuration
active_speaker_detection:
  enabled: true
  model_type: "lightweight"  # or "advanced"
  
  # Lightweight ASD settings
  audio_window_ms: 500
  motion_threshold: 0.1
  confidence_threshold: 0.6
  temporal_smoothing: 0.8
  
  # Advanced ASD settings
  model_path: "models/talknet_asd.onnx"
  feature_dim: 512
  attention_heads: 8
  temporal_context_frames: 10
  
  # Integration settings
  fallback_to_face_locking: true
  combine_with_face_confidence: true
  speaker_change_threshold: 0.3
  
  # Debug options
  visualize_speaker_scores: false
  save_audio_features: false
  log_speaker_changes: true
```

### **Success Criteria**
- [ ] <50ms additional latency for ASD processing
- [ ] Comprehensive YAML configuration system
- [ ] Robust error handling with graceful fallbacks
- [ ] Production-ready monitoring and debugging tools
- [ ] Seamless integration with existing MuseTalk workflow

---

## 🎯 Phase 4: Advanced Features & Multi-Speaker Scenarios

**Duration**: 3-4 days  
**Risk Level**: High  
**Goal**: Handle complex scenarios like speaker changes, overlapping speech, and group conversations

### **Objectives**
- Implement dynamic speaker switching for conversation scenarios
- Add support for overlapping speech detection
- Create group conversation handling with multiple active speakers
- Implement speaker identity tracking across video sequences

### **Key Actions**
1. **Dynamic Speaker Switching**
   - Implement smooth transitions between speakers
   - Add hysteresis to prevent rapid speaker switching
   - Create speaker change detection with confidence scoring
   - Implement temporal consistency for speaker identity

2. **Overlapping Speech Handling**
   - Detect multiple simultaneous speakers
   - Implement priority-based speaker selection
   - Add support for dominant speaker identification
   - Create blended lip-sync for multiple speakers

3. **Group Conversation Support**
   - Handle videos with 3+ people speaking
   - Implement conversation flow analysis
   - Add speaker turn prediction
   - Create intelligent camera focus following

4. **Speaker Identity Tracking**
   - Implement speaker embedding for identity consistency
   - Add face-voice association learning
   - Create speaker re-identification across cuts
   - Implement speaker profile persistence

### **Advanced Features**
```python
class MultiSpeakerASD:
    def __init__(self):
        self.speaker_tracker = SpeakerIdentityTracker()
        self.conversation_analyzer = ConversationFlowAnalyzer()
        self.transition_smoother = SpeakerTransitionSmoother()
        
    def handle_multi_speaker_scenario(self, faces, audio, conversation_context):
        # Detect all active speakers
        active_speakers = self.detect_multiple_speakers(faces, audio)
        
        # Analyze conversation flow
        conversation_state = self.conversation_analyzer.analyze(active_speakers, conversation_context)
        
        # Select primary speaker based on context
        primary_speaker = self.select_primary_speaker(active_speakers, conversation_state)
        
        # Smooth transitions
        final_speaker = self.transition_smoother.apply(primary_speaker, conversation_context)
        
        return final_speaker
```

### **Success Criteria**
- [ ] Smooth speaker transitions without jarring switches
- [ ] Accurate handling of overlapping speech scenarios
- [ ] Support for group conversations with 3+ speakers
- [ ] Consistent speaker identity tracking across video
- [ ] Intelligent conversation flow analysis

---

## 📊 Technical Implementation Details

### **File Modifications Required**
- `app/braivtalk/utils/face_detection/api.py` - Add ASD integration to face selection
- `app/braivtalk/utils/preprocessing.py` - Add audio processing pipeline
- `app/scripts/inference.py` - Add ASD configuration and initialization
- `app/braivtalk/utils/audio_processing.py` - **NEW** Audio feature extraction
- `app/braivtalk/utils/speaker_detection.py` - **NEW** ASD model implementations
- `configs/inference/test.yaml` - Add ASD configuration parameters

### **New Dependencies**
```python
# Audio processing
librosa>=0.9.0
soundfile>=0.10.0
webrtcvad>=2.0.10

# Machine learning
torch>=1.12.0
torchaudio>=0.12.0
onnxruntime-gpu>=1.12.0

# Feature extraction
opencv-python>=4.6.0
scikit-learn>=1.1.0
```

### **Model Requirements**
- **Lightweight ASD**: No additional models (uses existing audio/visual processing)
- **Advanced ASD**: TalkNet or similar pre-trained model (~50MB ONNX file)
- **Audio Features**: Optional pre-trained audio encoder (~20MB)
- **Visual Features**: Uses existing YOLOv8 face features

### **Configuration Integration**
```yaml
# Add to existing test.yaml
active_speaker_detection:
  enabled: true
  model_type: "lightweight"  # "lightweight" or "advanced"
  
  # Audio processing
  sample_rate: 16000
  audio_window_ms: 500
  voice_activity_threshold: 0.01
  
  # Visual processing  
  motion_threshold: 0.1
  landmark_smoothing: 0.8
  
  # Fusion parameters
  audio_weight: 0.6
  visual_weight: 0.4
  confidence_threshold: 0.6
  temporal_smoothing: 0.8
  
  # Integration with face selection
  combine_with_face_confidence: true
  fallback_to_face_locking: true
  speaker_change_hysteresis: 0.2
  
  # Debug options
  debug_speaker_detection: false
  save_speaker_timeline: false
  visualize_audio_energy: false
```

---

## 🎯 Success Metrics

### **Performance Targets**
- **Accuracy**: 90%+ speaker detection accuracy on multi-speaker videos
- **Speed**: <50ms additional latency for ASD processing
- **Robustness**: 95%+ uptime with graceful fallback handling
- **Quality**: Elimination of face switching issues in 95%+ of cases

### **Quality Assessments**
- Multi-speaker video test suite with ground truth labels
- Quantitative speaker detection accuracy measurements
- User acceptance testing for natural speaker transitions
- Performance benchmarking against baseline face selection

### **Test Scenarios**
1. **Single Speaker**: Verify no regression from current system
2. **Speaker Changes**: Test smooth transitions between speakers
3. **Overlapping Speech**: Handle multiple simultaneous speakers
4. **Group Conversations**: 3+ people with natural conversation flow
5. **Noisy Audio**: Robust performance with background noise
6. **Poor Lighting**: Visual motion detection in challenging conditions

---

## 🚨 Risk Mitigation

### **High-Risk Items**
1. **Model Availability** - Ensure pre-trained ASD models are accessible
2. **Audio-Video Sync** - Handle potential synchronization issues
3. **Performance Impact** - Maintain real-time processing speeds
4. **Integration Complexity** - Seamless integration with existing pipeline

### **Contingency Plans**
- Keep existing face locking as fallback option
- Implement gradual rollout with A/B testing
- Create comprehensive test suite for validation
- Maintain model-agnostic architecture for easy swapping

### **Fallback Strategies**
- **ASD Failure**: Automatic fallback to primary face locking
- **Audio Issues**: Use visual-only motion detection
- **Performance Issues**: Dynamic quality reduction
- **Model Loading Errors**: Graceful degradation to existing system

---

## 📅 Timeline Summary

| Phase | Duration | Risk | Key Deliverable | Dependencies |
|-------|----------|------|-----------------|--------------|
| Phase 0 | 1-2 days | Very Low | Face detection debug & visualization tool | Existing YOLOv8 pipeline |
| Phase 1 | 2-3 days | Low | Lightweight audio-visual ASD | Audio processing pipeline |
| Phase 2 | 4-5 days | Medium | Advanced TalkNet-style ASD | Pre-trained ASD models |
| Phase 3 | 2-3 days | Low | Production optimization | Performance profiling |
| Phase 4 | 3-4 days | High | Multi-speaker scenarios | Advanced conversation analysis |
| **Total** | **12-17 days** | | **Complete ASD integration** | **Full audio-visual pipeline** |

---

## 🔄 Integration with Existing System

### **Current State (YOLOv8 + Face Locking)**
```python
# Current face selection logic
selected_face = yolo_detector.select_best_face(faces, confidences, landmarks)
apply_ai_mouth(frame, selected_face)
```

### **Enhanced State (YOLOv8 + ASD)**
```python
# Enhanced face selection with ASD
audio_segment = extract_audio_for_frame(frame_idx)
speaker_scores = asd_detector.detect_active_speaker(faces, audio_segment)
selected_face = yolo_detector.select_best_face_with_asd(faces, confidences, speaker_scores)
apply_ai_mouth(frame, selected_face)
```

### **Backward Compatibility**
- ASD can be disabled via configuration
- Automatic fallback when ASD is uncertain
- Existing face locking remains as backup system
- No changes to output format or quality

---

*This plan provides a structured approach to implementing Active Speaker Detection for intelligent face selection in multi-speaker scenarios, while maintaining compatibility with the existing YOLOv8-enhanced MuseTalk pipeline.*
