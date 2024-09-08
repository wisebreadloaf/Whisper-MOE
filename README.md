## Whisper MOE 8X241M

A generalist model that is designed to perform various cross-attention tasks simultaneously, including text-to-speech, speech-to-text, translation, transcription, ASL to language conversion, and more. It incorporates several advanced techniques from papers like Test Time Training (TTT), Whisper, Patch-N-Pack, Mixture of Experts (MoE), and Grokking. The training approach involves fine-tuning eight versions of the Whisper model, each augmented with a TTT MLP backbone. The model uses fixed-size embeddings for inputs and outputs, packs them into a 2048-token context length, and employs an MOE router to dynamically select two experts per forward pass. This architecture will achieve higher performance than models of similar size due to its efficient parameter usage, allowing it to handle large-scale datasets effectively while being adaptable to various tasks. Due to the usage of Mixture of experts, the model will also be a life long learner.

## Input Modalities

- **Text:**

  - Input format: Plain text
  - Preprocessing steps: Tokenization, embedding, positional encoding
  - Dimensions: Sequence length, embedding size

- **Audio:**

  - Input format: m4a file format
  - Preprocessing steps: removal of channel dimension, long mel spectrogram, normalisation
  - Dimensions: Channels, Frames, Features

- **Video:**

  - Input format: RGB, MOV file format
  - Preprocessing steps: Patchify, flatten, positional encoding
  - Dimensions: frames ,Height, width, channels

## Output Modalities

- **Text:**

  - Output format: embedding
  - Post-processing steps: decoding, detokenization
  - Dimensions: Sequence length, embedding size

- **Audio:**

  - Input format: embedding
  - Post-processing steps: decoding, de-normalisation, long mel spectrogram, addition channel dimension
  - Dimensions: Channels, Frames, Features

## Architecture References

- **Base Architecture:**

  - Model type: Transformer, Test Time Training(MLP)
  - Key components: Cross Attention mechanism, convolution layers, Router.
  - Reference papers: Provide citation or link to the original paper.
  
    i) whisper(https://cdn.openai.com/papers/whisper.pdf)
    </br>
    ii) TTT(https://arxiv.org/pdf/2407.04620)
    </br>
    iii) Grokking(https://arxiv.org/pdf/2201.02177)
    </br>
    iv) MoE(https://arxiv.org/pdf/2401.04088)
    </br>
    v) Patch-N-Pack(https://arxiv.org/pdf/2307.06304)
    </br>
    vi) Generalist agent(https://arxiv.org/pdf/2205.06175)

- **Architectural Variants:**
  - Variants: conformers are a variant of transformers which combine transformers and CNN's
  - Modifications: Addition of TTT(MLP) layer as the backbone of transformer model.
  - Use cases: Life long learning, Parameter efficient, Multi-task learning, Long context.

## Training Configuration

- **Data Requirements:**

  - Datasets: List of datasets used for training.
    
    i) [Kathbath](https://huggingface.co/datasets/ai4bharat/kathbath)
    </br>
    ii) [Include](https://huggingface.co/datasets/ai4bharat/INCLUDE)
    </br>
    iii) [Wiki-translate](https://huggingface.co/datasets/ai4bharat/wiki-translate)

  - Data preprocessing: Steps and tools used for preprocessing.
    i) Datasets python module
    ii) Typing python module
    iii) Dataclasses python module

- **Training Hyperparameters:**

  - learning_rate: 1.7e-05
  - train_batch_size: 32
  - eval_batch_size: 32
  - seed: 42
  - optimizer: adamw with betas=(0.9, 0.98) and epsilon=1e-06
  - lr_scheduler_type: linear
  - lr_scheduler_warmup_steps: 10000
  - training_steps: 5000
  - mixed_precision_training: true

- **Training Environment:**

  - Hardware: GPUs
  - Framework: PyTorch
  - Distributed training: Data parallelism, model parallelism

## Evaluation Metrics

- **Primary Metrics:**

  - Metric 1: CE loss
  - Metric 2: Word error rate
  - Metric 3: Character Error rate

- **Secondary Metrics:**
  - Metric 1: Latency, throughput.
  - Metric 2: Model size, FLOPs.

## Deployment Considerations

- **Inference Optimization:**

  - Techniques: Quantization, pruning
  - Deployment platform: Cloud

- **Scalability:**
  - Horizontal scaling: e.g., Multi-GPU inference, distributed inference.

## Versioning and Maintenance

- **Version Control:**

  - Git repository: Link to the repository.
  - Version tagging: v1.0, v2.0, etc.

- **Maintenance Plan:**
  - Scheduled updates: Frequency of updates and improvementsAddition of more experts, Mixture of million experts, Splitting up each expert into multiple experts(Deep seek MOE).
  - Bug fixes: Process for identifying and resolving issues.
  - Documentation: Keeping the documentation up-to-date.
