# NeuroFocus Project Report

## What is this project about?
This project helps determine if a person is paying attention by looking at their brain signals (EEG) and eye state (open/closed). Think of it like a smart system that can tell if you're focused or distracted!

## Understanding Brain Signals

### What are Brain Waves?
Brain waves are patterns of electrical activity in your brain. Just like ocean waves can be big or small, brain waves have different sizes and speeds:

1. **Beta Waves (13-30 Hz)**
   - Fast waves
   - Present when you're alert and focused
   - Like when you're solving a math problem

2. **Alpha Waves (8-13 Hz)**
   - Medium-speed waves
   - Show up when you're relaxed but awake
   - Like when you're sitting quietly with eyes closed

3. **Theta Waves (4-8 Hz)**
   - Slower waves
   - Appear when you're drowsy or daydreaming
   - Common during light sleep

4. **Delta Waves (0.5-4 Hz)**
   - The slowest waves
   - Mainly during deep sleep
   - Not used much in our attention detection

### What are Artifacts?
Artifacts are unwanted signals that mix with brain waves, like noise in a radio signal. Common types:

1. **Eye Blinks**
   - Cause big spikes in the signal
   - Happen every few seconds
   - Can look like attention changes but aren't

2. **Muscle Movements**
   - Cause irregular patterns
   - From jaw clenching, neck movement, etc.
   - Can confuse the attention detection

## The Dataset
We use the **UCI EEG Eye State Dataset** which contains:
- Brain signals from 14 different points on the head
- Information about whether the eyes are open or closed
- About 15,000 samples of data
- Each sample is labeled as either:
  - 0 = Eyes Closed
  - 1 = Eyes Open

## How do we calculate attention?
We don't just look at whether the eyes are open or closed. Here's how we figure out if someone is paying attention:

1. **Brain Waves Analysis**: 
   - Look for strong Beta waves (sign of focus)
   - Check if Alpha waves are low (less relaxation)
   - Monitor Theta waves (check for drowsiness)

2. **The Magic Ratio**: 
   - We calculate: Beta waves ÷ Alpha waves
   - Higher ratio = Better attention
   - Lower ratio = Less attention

3. **Combined Analysis**:
   - Eyes open + High Beta/Alpha ratio = Good attention
   - Eyes open + Low Beta/Alpha ratio = Poor attention
   - Eyes closed = Not paying attention

## Simulation of Brain Signals
Sometimes we need practice data to test our system. Here's how we create it:

### How We Generate Signals
1. **Base Waves**
   - Create artificial Beta, Alpha, and Theta waves
   - Mix them in different amounts
   - Add small random variations to make them realistic

2. **Attention States**
   - Focused State:
     - Strong Beta waves
     - Weak Alpha waves
     - Very weak Theta waves
   - Unfocused State:
     - Weak Beta waves
     - Strong Alpha waves
     - Medium Theta waves

3. **Adding Artifacts**
   - Eye Blinks: Add spikes every 5 seconds
   - Muscle Movement: Add random bursts of high-frequency noise
   - Natural Noise: Add tiny random variations

### Impact of Artifacts
- Can make attention detection less accurate
- System needs to learn to ignore them
- Some artifacts (like eye blinks) can actually help detect attention

## Our Models
We built three different AI models to analyze this data:

### 1. EEGNet
- Like a specialized brain signal detector
- Architecture:
  ```
  Input (14 channels)
  │
  ├─> Temporal Convolution (8 filters)
  │   └─> Batch Normalization
  │
  ├─> Spatial Convolution (16 filters)
  │   └─> Batch Normalization
  │
  ├─> Separable Convolution (16 filters)
  │   └─> Batch Normalization
  │
  ├─> Average Pooling
  │
  └─> Fully Connected Layer (2 outputs)
  ```
- Best for accurate but slower analysis

### 2. SimpleCNN
- A lighter, faster model
- Architecture:
  ```
  Input (14 channels)
  │
  ├─> Convolution Layer 1 (32 filters)
  │   ├─> Batch Normalization
  │   └─> ReLU Activation
  │
  ├─> Convolution Layer 2 (64 filters)
  │   ├─> Batch Normalization
  │   └─> ReLU Activation
  │
  ├─> Adaptive Max Pooling
  │
  ├─> Fully Connected Layer 1 (64 units)
  │   └─> ReLU Activation
  │
  └─> Fully Connected Layer 2 (2 outputs)
  ```
- Good for real-time analysis

### 3. EEGMLP
- The simplest model
- Architecture:
  ```
  Input Layer
  │
  ├─> Hidden Layer 1 (256 units)
  │   ├─> ReLU Activation
  │   └─> Dropout (30%)
  │
  ├─> Hidden Layer 2 (128 units)
  │   ├─> ReLU Activation
  │   └─> Dropout (30%)
  │
  ├─> Hidden Layer 3 (64 units)
  │   ├─> ReLU Activation
  │   └─> Dropout (30%)
  │
  └─> Output Layer (2 units)
  ```
- Easy to adjust and train

## Comparing the Models

Think of our three models like different types of doctors examining brain activity:

### Model Comparison
```
Model     | Accuracy | Speed | Complexity
----------|----------|-------|------------
EEGNet    | Highest  | Slow  | High
SimpleCNN | Good     | Fast  | Medium
EEGMLP    | Basic    | Rapid | Low
```

### Key Differences

1. **EEGNet**
   - Like a brain specialist who knows exactly what to look for
   - **Advantages**:
     - Most accurate for brain signals
     - Best at understanding relationships between brain regions
     - Catches subtle patterns in the data
   - **Disadvantages**:
     - Takes more time to process
     - Needs more powerful computers
     - More complex to set up

2. **SimpleCNN**
   - Like a general doctor who can spot most issues quickly
   - **Advantages**:
     - Good balance of speed and accuracy
     - Works well for real-time analysis
     - Reliable for most situations
   - **Disadvantages**:
     - Not as specialized for brain signals
     - Might miss some complex patterns

3. **EEGMLP**
   - Like a quick health screening - fast but basic
   - **Advantages**:
     - Fastest processing
     - Easiest to train and modify
     - Works on simple computers
   - **Disadvantages**:
     - Most basic analysis
     - Needs more data to learn well
     - May miss important patterns

### When to Use Each Model?
- Need highest accuracy? → Use EEGNet
- Need real-time processing? → Use SimpleCNN
- Need something simple and fast? → Use EEGMLP

## Results
- The system is about 85-90% accurate in detecting attention
- Can work in real-time
- Can handle noise and interference in the brain signals

## Why is this useful?
- Can help students monitor their attention during study
- Could be used in classrooms to check student engagement
- Might help people improve their focus
- Could be useful for attention disorder research

## Simple Example
Imagine you're reading a book:
1. The system checks if your eyes are open (basic attention)
2. It then looks at your brain waves to see if you're actually focusing on the book
3. Even if your eyes are open, if your brain waves show low attention, it knows you're not really focused
4. This gives a much better picture than just checking if someone's eyes are open!

## Future Improvements
1. Make it work on phones
2. Make it faster
3. Make it more accurate
4. Add more types of measurements
5. Make it easier to use 

## Team Members
1. Sri Kaushik Kesanapalli - AM.EN.U4AIE22026
2. Mahin S Baiju - AM.EN.U4EAC22039
3. Indiresh Sunkara - AM.EN.U4ECE22044
4. Siddharth R - AM.EN.U4ECE22042
