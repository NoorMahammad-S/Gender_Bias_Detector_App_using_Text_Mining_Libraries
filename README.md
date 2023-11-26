# Gender Bias Detector App using Text Mining Libraries

## Overview

This Python script demonstrates the creation of a Gender Bias Detector using text mining libraries such as NLTK and scikit-learn. The detector predicts the likely gender associated with a given name based on the last letter of the name.

## Requirements

Ensure you have Python installed. You can install the required dependencies using:

```bash
pip install -r requirements.txt
```

## Usage

1. Clone the repository:

```bash
git clone https://github.com/NoorMahammad-S/Gender_Bias_Detector_App_using_Text_Mining_Libraries.git
cd Gender_Bias_Detector_App_using_Text_Mining_Libraries
```

2. Run the Gender Bias Detector script:

```bash
python main.py
```

3. The script will train a Naive Bayes classifier, evaluate its accuracy, and provide gender predictions for a list of names.

## Files

- **main.py**: The main script containing the Gender Bias Detector implementation.
- **requirements.txt**: Lists the required Python libraries and their versions.

## Example

To detect gender bias in custom text, use the `detect_gender_bias` function:

```python
from gender_bias_detector import detect_gender_bias

text_to_analyze = "The nurse brought the medication to the patient."
detected_gender = detect_gender_bias(text_to_analyze)
print(f"Detected gender bias: {detected_gender}")
```

