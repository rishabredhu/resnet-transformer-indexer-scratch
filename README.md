
# Resume Parser with Named Entity Recognition (NER) and ResNet-Based Visual Feature Extraction

This project implements a resume parser using Named Entity Recognition (NER) with state-of-the-art deep learning techniques, specifically leveraging the [RoBERTa transformer model](https://arxiv.org/abs/1907.11692) for extracting key textual entities from resumes. Additionally, it incorporates a custom-built **ResNet** model from scratch to extract visual features from resumes, enabling more comprehensive document understanding by combining text and visual layout information.

> **Note:** This is a personal project and is not production-ready. It can perform decently for parsing resumes and extracting keywords to match with relevant entities. The ResNet model provides additional visual insights, useful in cases where document layout matters.

## Table of Contents

- [Features](#features)
- [Entity Types](#entity-types)
- [Technologies Used](#technologies-used)
- [ResNet Implementation](#resnet-implementation)
- [Setup](#setup)
- [Usage](#usage)
- [Future Work](#future-work)
- [License](#license)

## Features

- Parses resumes to extract key details using Named Entity Recognition (NER).
- Uses the **RoBERTa** transformer model to identify and classify text-based entities.
- Implements a **ResNet** model from scratch to extract visual features from resume images, enhancing the analysis of document structure and layout.
- Matches extracted keywords with predefined entities.

## Entity Types

The parser identifies and classifies words into the following seven entities:

1. **Job Title**: Type of job the user is looking for.
2. **Skill**: Important skills possessed by the user.
3. **Experience**: Previous job positions and their timelines.
4. **Org**: Companies the user has worked with or is currently working at.
5. **Tool**: Software tools used by the user.
6. **Degree**: Educational qualifications (e.g., B.Tech, M.Tech, MBA).
7. **Educ**: Institutions where the user has studied.

## Technologies Used

- **RoBERTa**: Transformer model for NER.
- **ResNet**: Custom-built convolutional neural network for visual feature extraction.
- **Python**: Main programming language.
- **Transformers**: Hugging Face library for working with RoBERTa.
- **SpaCy**: For NER and NLP processing.
- **PyTorch**: For building and training the ResNet model.
- **FAISS**: For efficient similarity-based matching (if required for future extensions).

## ResNet Implementation

To complement the textual features extracted by the RoBERTa model, this project also includes a custom implementation of **ResNet** (Residual Network) for extracting visual features from resume images. The ResNet model was built from scratch using **PyTorch** to capture document layout, font types, and other visual cues that might help in better understanding resume content.

### Key Features of ResNet Implementation:
- **Residual Connections**: Helps mitigate the vanishing gradient problem and allows for deeper architectures.
- **Feature Extraction**: Converts resume images into visual embeddings that can be compared for similarity-based searches.
- **Pretrained Option**: Includes both a custom-trained model and the option to load pretrained weights from **ResNet-50** for more advanced use cases.

## Setup

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/resume-parser-ner.git
    cd resume-parser-ner
    ```

2. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

3. Download the pretrained RoBERTa model using Hugging Face:

    ```bash
    from transformers import RobertaForTokenClassification, RobertaTokenizer

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model = RobertaForTokenClassification.from_pretrained("roberta-base")
    ```

4. For the ResNet model, either build from scratch or use the pretrained ResNet-50:

    ```bash
    # PyTorch code to load or build the ResNet model
    import torch
    from torchvision import models

    # Option 1: Pretrained ResNet-50
    model = models.resnet50(pretrained=True)

    # Option 2: Build ResNet from scratch (simplified version)
    class SimpleResNet(torch.nn.Module):
        # Define ResNet architecture here...
        pass
    ```

## Usage

1. Add resumes (in text or PDF format) to the appropriate directory.
2. Convert PDFs to images (if necessary) for visual feature extraction using ResNet.
3. Run the parser script to extract entities:

    ```bash
    python parse_resume.py --input path_to_resume_file
    ```

4. The output will show extracted entities (e.g., job titles, skills, etc.) and visual features for similarity matching.

## Future Work

- Improve entity classification accuracy with more advanced deep learning techniques.
- Optimize the ResNet model for faster visual feature extraction.
- Combine text and visual embeddings for more accurate similarity-based searches.
- Integrate support for multi-language resumes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Let me know if you need further adjustments or additional sections!
