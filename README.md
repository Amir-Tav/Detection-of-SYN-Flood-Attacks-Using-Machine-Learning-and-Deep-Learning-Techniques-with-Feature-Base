# Classifying SYN Flood Attacks from Normal Network Traffic Using ML and DL Methods

---

## Overview

This project focuses on classifying SYN flood attacks from normal network traffic using various machine learning (ML) and deep learning (DL) techniques. Our goal is to create efficient, real-time capable models that can distinguish SYN flood patterns from benign traffic behavior.

## Dataset: CIC-DDoS2019

We use the [CIC-DDoS2019 dataset](https://www.unb.ca/cic/datasets/ddos-2019.html) provided by the Canadian Institute for Cybersecurity (CIC). 

**Key Points about the Dataset:**
- Simulates real-world DDoS attacks in a realistic network environment.
- Traffic generated from diverse attack types, including SYN Flood, UDP Flood, and HTTP Flood.
- Covers different attack scenarios across multiple days.
- Contains detailed flow-level features (over 85 per flow) extracted using CICFlowMeter.
- Includes timestamps, flow duration, packet sizes, and flag counts.

For this project, we specifically extract SYN flood attack samples and benign traffic for a targeted binary classification task.

## Research Question

**Can we classify SYN flood attacks from normal network traffic using various ML and DL methods?**

## Pre-processing

Given the large size and complexity of the original dataset, we performed focused preprocessing steps to prepare the data for efficient real-time prediction:

- **Feature Selection:** We selected 13 features highly relevant to SYN flood detection, balancing informativeness and computational efficiency:
  1. **SYN Flag Count:** Identifies SYN packets, critical to this attack type.
  2. **Total Fwd Packets:** High counts can indicate flooding attempts.
  3. **Total Backward Packets:** Low response suggests incomplete handshakes.
  4. **Flow Duration:** Short flows are typical of SYN floods.
  5. **Flow Packets/s:** Rapid packet rates can signal an attack.
  6. **Flow Bytes/s:** Highlights volume and pattern of transmitted data.
  7. **Fwd Packet Length Mean:** Uniform packet sizes are common in SYN floods.
  8. **Bwd Packet Length Mean:** Few or no backward packets suggest missing responses.
  9. **Bwd IAT Mean:** Irregular reply intervals may indicate an attack.
  10. **ACK Flag Count:** Low ACK counts imply incomplete TCP handshakes.
  11. **Active Mean:** Short active times are characteristic of flooding.
  12. **Inbound:** Directionality helps identify the attack target.
  13. **Label:** Specifies whether the flow is benign or SYN attack.

- **Sampling:** We randomly selected 5,000 SYN attack instances and 5,000 benign instances, creating a balanced and manageable 10,000-row dataset.

- **Cleaning:** We removed rows containing null (NA) or infinite (Inf) values to ensure reliable and error-free model training.

---

This structured preprocessing ensures a clean, efficient, and representative dataset ready for training robust ML and DL models for SYN flood detection.
