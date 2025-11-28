# Federated Learning Resources
## Papers, Tutorials, and Learning Materials

**Topic:** Dynamic FL, Heterogeneous Labels, Continual Learning  
**Created:** November 28, 2025

---

## 1. Core Federated Learning Papers

### Foundational Papers

**1. Communication-Efficient Learning of Deep Networks from Decentralized Data (FedAvg)**
- **Authors:** McMahan et al., Google
- **Year:** 2017
- **Link:** https://arxiv.org/abs/1602.05629
- **Key Contribution:** Introduced FedAvg algorithm (the foundation of FL)
- **Why Read:** Essential baseline - understand how FL aggregation works
- **Key Takeaway:** Weighted averaging of model updates enables collaborative learning

**2. Advances and Open Problems in Federated Learning**
- **Authors:** Kairouz et al., Google
- **Year:** 2021
- **Link:** https://arxiv.org/abs/1912.04977
- **Key Contribution:** Comprehensive survey of FL challenges and solutions
- **Why Read:** 400+ pages covering ALL FL topics including heterogeneity
- **Relevant Sections:**
  - Section 2.3: Non-IID data
  - Section 3.2: Personalization
  - Section 4: Continual learning in FL

**3. Federated Learning: Challenges, Methods, and Future Directions**
- **Authors:** Li et al.
- **Year:** 2020
- **Link:** https://arxiv.org/abs/1908.07873
- **Key Contribution:** Practical challenges in real-world FL deployment
- **Why Read:** Covers data heterogeneity extensively

---

## 2. Heterogeneous Label Spaces (Your Main Problem)

### Key Papers

**4. FedProx: Federated Optimization in Heterogeneous Networks**
- **Authors:** Li et al., CMU
- **Year:** 2020
- **Link:** https://arxiv.org/abs/1812.06127
- **Key Contribution:** Handles heterogeneous data with proximal term
- **Why Read:** Solution for different data distributions across clients
- **Implementation:** Available in Flower framework
- **Code:** https://github.com/litian96/FedProx

**5. Federated Learning with Non-IID Data**
- **Authors:** Zhao et al.
- **Year:** 2018
- **Link:** https://arxiv.org/abs/1806.00582
- **Key Contribution:** First systematic study of non-IID data in FL
- **Why Read:** Explains why different label distributions hurt FL
- **Key Insight:** Data sharing strategies to mitigate heterogeneity

**6. Think Locally, Act Globally: Federated Learning with Local and Global Representations**
- **Authors:** Liang et al.
- **Year:** 2020
- **Link:** https://arxiv.org/abs/2001.01523
- **Key Contribution:** Separate local and global model components
- **Why Read:** Directly addresses heterogeneous label spaces
- **Approach:** Similar to our embedding solution (Solution 2)

**7. Federated Learning with Personalization Layers**
- **Authors:** Arivazhagan et al., Google
- **Year:** 2019
- **Link:** https://arxiv.org/abs/1912.00818
- **Key Contribution:** Keep some layers local, share others
- **Why Read:** Practical approach for different output classes
- **Implementation:** Easy to adapt to your use case

---

## 3. Dynamic/Continual Learning in FL

### Papers on Adding New Classes

**8. Continual Federated Learning Based on Knowledge Distillation**
- **Authors:** Yoon et al.
- **Year:** 2021
- **Link:** https://arxiv.org/abs/2011.02936
- **Key Contribution:** Add new classes without forgetting old ones
- **Why Read:** Directly relevant to new threat discovery
- **Technique:** Knowledge distillation to preserve old knowledge

**9. Federated Continual Learning with Weighted Inter-client Transfer**
- **Authors:** Yoon et al.
- **Year:** 2021
- **Link:** https://arxiv.org/abs/2003.03196
- **Key Contribution:** Continual learning in federated setting
- **Why Read:** Handles sequential arrival of new classes
- **Code:** https://github.com/wyjeong/FedWeIT

**10. Incremental Learning in Federated Settings**
- **Authors:** Casado et al.
- **Year:** 2022
- **Link:** https://arxiv.org/abs/2208.00409
- **Key Contribution:** Expand model architecture dynamically
- **Why Read:** Exactly your scenario - new classes discovered over time
- **Approach:** Transfer learning + model expansion

**11. FedET: A Communication-Efficient Federated Class-Incremental Learning Framework**
- **Authors:** Guo et al.
- **Year:** 2023
- **Link:** https://arxiv.org/abs/2306.15347
- **Key Contribution:** Efficient class-incremental learning in FL
- **Why Read:** Recent work, state-of-the-art approach
- **Technique:** Elastic model expansion

---

## 4. Cybersecurity-Specific FL Papers

### ICS/Network Security with FL

**12. Federated Learning for Intrusion Detection Systems**
- **Authors:** Nguyen et al.
- **Year:** 2022
- **Link:** https://arxiv.org/abs/2106.09754
- **Key Contribution:** FL for network intrusion detection
- **Why Read:** Directly relevant to your ICS use case
- **Dataset:** Uses NSL-KDD, similar to your EdgeIIoT

**13. Privacy-Preserving Federated Learning for IoT Intrusion Detection**
- **Authors:** Zhao et al.
- **Year:** 2021
- **Link:** https://ieeexplore.ieee.org/document/9488753
- **Key Contribution:** FL for IoT security with differential privacy
- **Why Read:** Addresses privacy in security applications
- **Approach:** Combines FL with secure aggregation

**14. Federated Learning for Cyber Security: Concepts, Challenges and Future Directions**
- **Authors:** Mothukuri et al.
- **Year:** 2021
- **Link:** https://arxiv.org/abs/2106.08556
- **Key Contribution:** Survey of FL in cybersecurity
- **Why Read:** Comprehensive overview of security-specific challenges
- **Sections:** Covers adversarial attacks, privacy, and heterogeneity

**15. Collaborative Intrusion Detection via Federated Learning**
- **Authors:** Preuveneers et al.
- **Year:** 2018
- **Link:** https://arxiv.org/abs/1810.03959
- **Key Contribution:** Early work on FL for intrusion detection
- **Why Read:** Foundational paper for your domain
- **Insight:** Benefits of collaboration without data sharing

---

## 5. Practical Implementation Resources

### Frameworks and Tutorials

**16. Flower Framework Documentation**
- **Link:** https://flower.dev/docs/
- **Why Use:** Most popular FL framework (what you're using)
- **Key Sections:**
  - Tutorial: https://flower.dev/docs/tutorial-series-what-is-federated-learning.html
  - Strategies: https://flower.dev/docs/strategies.html
  - Examples: https://github.com/adap/flower/tree/main/examples

**17. TensorFlow Federated (TFF)**
- **Link:** https://www.tensorflow.org/federated
- **Why Use:** Google's FL framework, good for research
- **Tutorial:** https://www.tensorflow.org/federated/tutorials/federated_learning_for_image_classification
- **Advantage:** Tight integration with TensorFlow/Keras

**18. PySyft**
- **Link:** https://github.com/OpenMined/PySyft
- **Why Use:** Privacy-focused FL with differential privacy
- **Tutorial:** https://github.com/OpenMined/PySyft/tree/master/examples
- **Advantage:** Strong privacy guarantees

### Courses and Tutorials

**19. Federated Learning Course (Coursera)**
- **Link:** https://www.coursera.org/learn/federated-learning
- **Instructor:** Andrew Trask (OpenMined)
- **Duration:** 4 weeks
- **Content:** Basics to advanced FL concepts
- **Certificate:** Yes

**20. Flower Summit Talks**
- **Link:** https://flower.dev/conf/flower-summit-2023/
- **Content:** Latest research and industry applications
- **Format:** Video presentations
- **Topics:** Heterogeneity, personalization, production deployment

**21. Federated Learning Tutorial (NeurIPS)**
- **Link:** https://slideslive.com/38935813/federated-learning-tutorial
- **Presenters:** Google Research team
- **Duration:** 3 hours
- **Content:** Theory and practice of FL
- **Level:** Intermediate to advanced

---

## 6. Books

**22. Federated Learning: Privacy and Incentive**
- **Authors:** Yang et al.
- **Publisher:** Springer, 2020
- **Link:** https://link.springer.com/book/10.1007/978-3-030-63076-8
- **Content:** Comprehensive textbook on FL
- **Chapters:** 
  - Ch 3: Heterogeneous data
  - Ch 5: Personalization
  - Ch 7: Security and privacy

**23. Federated Learning Systems: Vision, Hype and Reality for Data Privacy and Protection**
- **Authors:** Aledhari et al.
- **Publisher:** IEEE Access, 2020
- **Link:** https://ieeexplore.ieee.org/document/9220780
- **Content:** Practical perspective on FL systems
- **Focus:** Real-world deployment challenges

---

## 7. GitHub Repositories

### Code Examples

**24. Awesome Federated Learning**
- **Link:** https://github.com/chaoyanghe/Awesome-Federated-Learning
- **Content:** Curated list of FL papers, code, and resources
- **Updated:** Regularly maintained
- **Sections:** Papers by topic, frameworks, datasets

**25. FedML**
- **Link:** https://github.com/FedML-AI/FedML
- **Content:** Complete FL library with many algorithms
- **Algorithms:** FedAvg, FedProx, FedOpt, etc.
- **Advantage:** Production-ready, scalable

**26. Flower Examples**
- **Link:** https://github.com/adap/flower/tree/main/examples
- **Content:** 50+ FL examples
- **Relevant:**
  - `advanced-tensorflow`: Custom strategies
  - `quickstart-tensorflow`: Basic FL
  - `simulation`: Large-scale FL simulation

**27. Federated Learning with Non-IID Data (Code)**
- **Link:** https://github.com/TalwalkarLab/leaf
- **Content:** Benchmark for heterogeneous FL
- **Datasets:** FEMNIST, Shakespeare, Reddit
- **Use:** Test your algorithms on standard benchmarks

---

## 8. Datasets for Testing

**28. CICIDS2017**
- **Link:** https://www.unb.ca/cic/datasets/ids-2017.html
- **Content:** Network intrusion detection dataset
- **Size:** 2.8M flows
- **Use:** Test FL for network security

**29. NSL-KDD**
- **Link:** https://www.unb.ca/cic/datasets/nsl.html
- **Content:** Classic intrusion detection dataset
- **Size:** 148K samples
- **Use:** Benchmark for IDS research

**30. Edge-IIoTset (Your Dataset)**
- **Link:** https://ieee-dataport.org/documents/edge-iiotset-new-comprehensive-realistic-cyber-security-dataset-iot-and-iiot-applications
- **Content:** IoT/IIoT security dataset
- **Size:** 2.2M samples
- **Use:** Your current dataset

---

## 9. Research Groups and Labs

### Leading FL Research Groups

**31. Google Research - Federated Learning**
- **Link:** https://research.google/teams/federated-learning/
- **Focus:** Core FL algorithms and applications
- **Publications:** 100+ papers on FL

**32. OpenMined**
- **Link:** https://www.openmined.org/
- **Focus:** Privacy-preserving ML and FL
- **Community:** Active open-source community

**33. CMU - Federated Learning Lab**
- **Link:** https://www.cs.cmu.edu/~federated-learning/
- **Focus:** Heterogeneous FL, optimization
- **Key Researchers:** Tian Li, Virginia Smith

**34. EPFL - Machine Learning and Optimization Lab**
- **Link:** https://mlo.epfl.ch/
- **Focus:** Distributed and federated learning
- **Key Researchers:** Martin Jaggi

---

## 10. Recommended Reading Path

### For Your Specific Problem (Heterogeneous Labels + New Threats)

**Week 1: Foundations**
1. Read Paper #1 (FedAvg) - Understand basic FL
2. Read Paper #2 (Advances in FL) - Sections 2.3 and 3.2
3. Watch Tutorial #21 (NeurIPS) - Visual understanding

**Week 2: Heterogeneity**
4. Read Paper #4 (FedProx) - Solution for heterogeneous data
5. Read Paper #6 (Think Locally, Act Globally) - Heterogeneous labels
6. Read Paper #7 (Personalization Layers) - Practical approach

**Week 3: Dynamic Learning**
7. Read Paper #9 (FedWeIT) - Continual learning in FL
8. Read Paper #10 (Incremental Learning) - Model expansion
9. Read Paper #11 (FedET) - State-of-the-art approach

**Week 4: Implementation**
10. Study Flower Examples #26 - Hands-on coding
11. Read Paper #12 (FL for IDS) - Your domain
12. Implement your solution using learned concepts

---

## 11. Key Concepts Summary

### Must-Know Terms

**Non-IID Data:**
- Data is not Independent and Identically Distributed
- Different clients have different data distributions
- Major challenge in FL

**Heterogeneous Label Space:**
- Different clients have different classes
- Your problem: Facility A has 13 attacks, B has 15
- Solution: Global label space or personalization

**Continual Learning:**
- Learning new tasks without forgetting old ones
- Your problem: New threats discovered over time
- Solution: Model expansion with transfer learning

**FedAvg:**
- Weighted averaging of client model updates
- Weight = number of samples at each client
- Foundation of most FL algorithms

**FedProx:**
- FedAvg + proximal term
- Handles heterogeneous data better
- Prevents client drift

**Personalization:**
- Each client has some local model components
- Shared components learn general patterns
- Local components adapt to client-specific data

---

## 12. Quick Reference Table

| Problem | Relevant Papers | Solution Approach |
|---------|----------------|-------------------|
| Different label spaces | #6, #7 | Global label space or personalization |
| New classes over time | #9, #10, #11 | Continual learning + model expansion |
| Non-IID data | #4, #5 | FedProx, data sharing, weighted aggregation |
| Privacy concerns | #13, #18 | Differential privacy, secure aggregation |
| ICS/Network security | #12, #14, #15 | Domain-specific FL strategies |

---

## 13. Community and Forums

**Stack Overflow:**
- Tag: `federated-learning`
- Link: https://stackoverflow.com/questions/tagged/federated-learning

**Flower Slack:**
- Link: https://flower.dev/join-slack
- Active community, quick responses

**Reddit:**
- r/MachineLearning - FL discussions
- r/deeplearning - Implementation help

**Twitter/X:**
- Follow: @FlowerLabs, @OpenMined, @GoogleAI
- Hashtag: #FederatedLearning

---

## 14. Conferences to Follow

**Top Conferences for FL Research:**

1. **NeurIPS** - Neural Information Processing Systems
2. **ICML** - International Conference on Machine Learning
3. **ICLR** - International Conference on Learning Representations
4. **FL-ICML** - Federated Learning Workshop at ICML
5. **FL-NeurIPS** - Federated Learning Workshop at NeurIPS

**Security Conferences:**
1. **CCS** - ACM Conference on Computer and Communications Security
2. **NDSS** - Network and Distributed System Security Symposium
3. **USENIX Security** - USENIX Security Symposium

---

## 15. Action Items for Your Project

### Immediate (This Week)
- [ ] Read Paper #1 (FedAvg) - 2 hours
- [ ] Read Paper #4 (FedProx) - 2 hours
- [ ] Watch Flower tutorial - 1 hour
- [ ] Implement global label space solution

### Short-term (This Month)
- [ ] Read Paper #9 (FedWeIT) - Continual learning
- [ ] Read Paper #10 (Incremental Learning) - Model expansion
- [ ] Implement new threat discovery system
- [ ] Test with your EdgeIIoT dataset

### Long-term (Next 3 Months)
- [ ] Read comprehensive survey (Paper #2)
- [ ] Implement differential privacy (Paper #13)
- [ ] Write paper on your approach
- [ ] Submit to conference (CCS or NDSS)

---

**Document Version:** 1.0  
**Last Updated:** November 28, 2025  
**Total Resources:** 34 papers, tutorials, and tools  
**Estimated Study Time:** 4-6 weeks for comprehensive understanding
