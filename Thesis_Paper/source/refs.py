"""Bibliography for the P2 thesis. Keys are cited in the content files as [@key].

Each entry: (bibtex type, fields). IEEE text is generated from the same fields, and
latex_support/references.bib is exported from here, so the Word and LaTeX versions agree.
"""

from __future__ import annotations

REFS: dict[str, tuple[str, dict[str, str]]] = {
    "song2025hlora": ("misc", dict(author="Shezheng Song and Hao Xu and Jun Ma and Shasha Li and Long Peng and Qian Wan and Xiaodong Liu and Jie Yu", title="How to Alleviate Catastrophic Forgetting in {LLMs} Finetuning? Hierarchical Layer-Wise and Element-Wise Regularization", howpublished="arXiv:2501.13669", year="2025")),
    "mccloskey1989": ("article", dict(author="Michael McCloskey and Neal J. Cohen", title="Catastrophic Interference in Connectionist Networks: The Sequential Learning Problem", journal="Psychology of Learning and Motivation", volume="24", pages="109--165", year="1989")),
    "french1999": ("article", dict(author="Robert M. French", title="Catastrophic Forgetting in Connectionist Networks", journal="Trends in Cognitive Sciences", volume="3", number="4", pages="128--135", year="1999")),
    "parisi2019": ("article", dict(author="German I. Parisi and Ronald Kemker and Jose L. Part and Christopher Kanan and Stefan Wermter", title="Continual Lifelong Learning with Neural Networks: A Review", journal="Neural Networks", volume="113", pages="54--71", year="2019")),
    "wang2024survey": ("article", dict(author="Liyuan Wang and Xingxing Zhang and Hang Su and Jun Zhu", title="A Comprehensive Survey of Continual Learning: Theory, Method and Application", journal="IEEE Transactions on Pattern Analysis and Machine Intelligence", volume="46", number="8", pages="5362--5383", year="2024")),
    "biesialska2020": ("inproceedings", dict(author="Magdalena Biesialska and Katarzyna Biesialska and Marta R. Costa-juss{\\`a}", title="Continual Lifelong Learning in Natural Language Processing: A Survey", booktitle="Proceedings of the 28th International Conference on Computational Linguistics (COLING)", pages="6523--6541", year="2020")),
    "shi2024survey": ("misc", dict(author="Haizhou Shi and others", title="Continual Learning of Large Language Models: A Comprehensive Survey", howpublished="arXiv:2404.16789", year="2024")),
    "wu2024survey": ("misc", dict(author="Tongtong Wu and Linhao Luo and Yuan-Fang Li and Shirui Pan and Thuy-Trang Vu and Gholamreza Haffari", title="Continual Learning for Large Language Models: A Survey", howpublished="arXiv:2402.01364", year="2024")),
    "kirkpatrick2017": ("article", dict(author="James Kirkpatrick and others", title="Overcoming Catastrophic Forgetting in Neural Networks", journal="Proceedings of the National Academy of Sciences", volume="114", number="13", pages="3521--3526", year="2017")),
    "zenke2017": ("inproceedings", dict(author="Friedemann Zenke and Ben Poole and Surya Ganguli", title="Continual Learning Through Synaptic Intelligence", booktitle="Proceedings of the 34th International Conference on Machine Learning (ICML)", pages="3987--3995", year="2017")),
    "aljundi2018": ("inproceedings", dict(author="Rahaf Aljundi and Francesca Babiloni and Mohamed Elhoseiny and Marcus Rohrbach and Tinne Tuytelaars", title="Memory Aware Synapses: Learning What (Not) to Forget", booktitle="Proceedings of the European Conference on Computer Vision (ECCV)", year="2018")),
    "li2018lwf": ("article", dict(author="Zhizhong Li and Derek Hoiem", title="Learning without Forgetting", journal="IEEE Transactions on Pattern Analysis and Machine Intelligence", volume="40", number="12", pages="2935--2947", year="2018")),
    "lopezpaz2017": ("inproceedings", dict(author="David Lopez-Paz and Marc'Aurelio Ranzato", title="Gradient Episodic Memory for Continual Learning", booktitle="Advances in Neural Information Processing Systems (NeurIPS)", year="2017")),
    "chaudhry2019": ("inproceedings", dict(author="Arslan Chaudhry and Marc'Aurelio Ranzato and Marcus Rohrbach and Mohamed Elhoseiny", title="Efficient Lifelong Learning with {A-GEM}", booktitle="International Conference on Learning Representations (ICLR)", year="2019")),
    "yu2020pcgrad": ("inproceedings", dict(author="Tianhe Yu and Saurabh Kumar and Abhishek Gupta and Sergey Levine and Karol Hausman and Chelsea Finn", title="Gradient Surgery for Multi-Task Learning", booktitle="Advances in Neural Information Processing Systems (NeurIPS)", year="2020")),
    "rolnick2019": ("inproceedings", dict(author="David Rolnick and Arun Ahuja and Jonathan Schwarz and Timothy Lillicrap and Gregory Wayne", title="Experience Replay for Continual Learning", booktitle="Advances in Neural Information Processing Systems (NeurIPS)", year="2019")),
    "rusu2016": ("misc", dict(author="Andrei A. Rusu and others", title="Progressive Neural Networks", howpublished="arXiv:1606.04671", year="2016")),
    "schwarz2018": ("inproceedings", dict(author="Jonathan Schwarz and others", title="Progress {\\&} Compress: A Scalable Framework for Continual Learning", booktitle="Proceedings of the 35th International Conference on Machine Learning (ICML)", year="2018")),
    "chen2020recadam": ("inproceedings", dict(author="Sanyuan Chen and Yutai Hou and Yiming Cui and Wanxiang Che and Ting Liu and Xiangzhan Yu", title="Recall and Learn: Fine-tuning Deep Pretrained Language Models with Less Forgetting", booktitle="Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP)", pages="7870--7881", year="2020")),
    "luo2023empirical": ("misc", dict(author="Yun Luo and Zhen Yang and Fandong Meng and Yafu Li and Jie Zhou and Yue Zhang", title="An Empirical Study of Catastrophic Forgetting in Large Language Models During Continual Fine-tuning", howpublished="arXiv:2308.08747", year="2023")),
    "sun2020lamol": ("inproceedings", dict(author="Fan-Keng Sun and Cheng-Hao Ho and Hung-Yi Lee", title="{LAMOL}: Language Modeling for Lifelong Language Learning", booktitle="International Conference on Learning Representations (ICLR)", year="2020")),
    "dautume2019": ("inproceedings", dict(author="Cyprien de Masson d'Autume and Sebastian Ruder and Lingpeng Kong and Dani Yogatama", title="Episodic Memory in Lifelong Language Learning", booktitle="Advances in Neural Information Processing Systems (NeurIPS)", year="2019")),
    "hu2022lora": ("inproceedings", dict(author="Edward J. Hu and Yelong Shen and Phillip Wallis and Zeyuan Allen-Zhu and Yuanzhi Li and Shean Wang and Lu Wang and Weizhu Chen", title="{LoRA}: Low-Rank Adaptation of Large Language Models", booktitle="International Conference on Learning Representations (ICLR)", year="2022")),
    "wang2023olora": ("inproceedings", dict(author="Xiao Wang and Tianze Chen and Qiming Ge and Han Xia and Rong Bao and Rui Zheng and Qi Zhang and Tao Gui and Xuanjing Huang", title="Orthogonal Subspace Learning for Language Model Continual Learning", booktitle="Findings of the Association for Computational Linguistics: EMNLP 2023", year="2023")),
    "ren2024ilora": ("misc", dict(author="Weijieying Ren and Xinlong Li and Lei Wang and Tianxiang Zhao and Wei Qin", title="Analyzing and Reducing Catastrophic Forgetting in Parameter Efficient Tuning", howpublished="arXiv:2402.18865", year="2024")),
    "liang2024inflora": ("inproceedings", dict(author="Yan-Shuo Liang and Wu-Jun Li", title="{InfLoRA}: Interference-Free Low-Rank Adaptation for Continual Learning", booktitle="Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)", year="2024")),
    "zheng2026ewclora": ("inproceedings", dict(author="Yaoyue Zheng and Yin Zhang and Joost van de Weijer and Gido M. van de Ven and Shaoyi Du and Xuetao Zhang and Zhiqiang Tian", title="Revisiting Weight Regularization for Low-Rank Continual Learning", booktitle="International Conference on Learning Representations (ICLR)", note="arXiv:2602.17559", year="2026")),
    "biderman2024": ("article", dict(author="Dan Biderman and others", title="{LoRA} Learns Less and Forgets Less", journal="Transactions on Machine Learning Research", year="2024")),
    "houlsby2019": ("inproceedings", dict(author="Neil Houlsby and others", title="Parameter-Efficient Transfer Learning for {NLP}", booktitle="Proceedings of the 36th International Conference on Machine Learning (ICML)", year="2019")),
    "schulman2017ppo": ("misc", dict(author="John Schulman and Filip Wolski and Prafulla Dhariwal and Alec Radford and Oleg Klimov", title="Proximal Policy Optimization Algorithms", howpublished="arXiv:1707.06347", year="2017")),
    "williams1992": ("article", dict(author="Ronald J. Williams", title="Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning", journal="Machine Learning", volume="8", pages="229--256", year="1992")),
    "ouyang2022": ("inproceedings", dict(author="Long Ouyang and others", title="Training Language Models to Follow Instructions with Human Feedback", booktitle="Advances in Neural Information Processing Systems (NeurIPS)", year="2022")),
    "ziegler2019": ("misc", dict(author="Daniel M. Ziegler and others", title="Fine-Tuning Language Models from Human Preferences", howpublished="arXiv:1909.08593", year="2019")),
    "christiano2017": ("inproceedings", dict(author="Paul F. Christiano and Jan Leike and Tom B. Brown and Miljan Martic and Shane Legg and Dario Amodei", title="Deep Reinforcement Learning from Human Preferences", booktitle="Advances in Neural Information Processing Systems (NeurIPS)", year="2017")),
    "rafailov2023": ("inproceedings", dict(author="Rafael Rafailov and others", title="Direct Preference Optimization: Your Language Model is Secretly a Reward Model", booktitle="Advances in Neural Information Processing Systems (NeurIPS)", year="2023")),
    "shao2024deepseekmath": ("misc", dict(author="Zhihong Shao and others", title="{DeepSeekMath}: Pushing the Limits of Mathematical Reasoning in Open Language Models", howpublished="arXiv:2402.03300", year="2024")),
    "deepseek2025r1": ("misc", dict(author="{DeepSeek-AI}", title="{DeepSeek-R1}: Incentivizing Reasoning Capability in {LLMs} via Reinforcement Learning", howpublished="arXiv:2501.12948", year="2025")),
    "lambert2024tulu": ("misc", dict(author="Nathan Lambert and others", title="{T\\\"ulu} 3: Pushing Frontiers in Open Language Model Post-Training", howpublished="arXiv:2411.15124", year="2024")),
    "kaplanis2019": ("inproceedings", dict(author="Christos Kaplanis and Murray Shanahan and Claudia Clopath", title="Policy Consolidation for Continual Reinforcement Learning", booktitle="Proceedings of the 36th International Conference on Machine Learning (ICML)", year="2019")),
    "shenfeld2025razor": ("misc", dict(author="Idan Shenfeld and Jyothish Pari and Pulkit Agrawal", title="{RL}'s Razor: Why Online Reinforcement Learning Forgets Less", howpublished="arXiv:2509.04259", year="2025")),
    "lai2025rft": ("misc", dict(author="Song Lai and others", title="Reinforcement Fine-Tuning Naturally Mitigates Forgetting in Continual Post-Training", howpublished="arXiv:2507.05386", year="2025")),
    "mukherjee2025sparse": ("misc", dict(author="Sagnik Mukherjee and Lifan Yuan and Dilek Hakkani-T{\\\"u}r and Hao Peng", title="Reinforcement Learning Finetunes Small Subnetworks in Large Language Models", howpublished="arXiv:2505.11711", year="2025")),
    "luo2026cpo": ("misc", dict(author="Mao-Lin Luo and others", title="{RL} Forgets! Towards Continual Policy Optimization", howpublished="arXiv:2607.04364", year="2026")),
    "wang2026geometry": ("misc", dict(author="Yuanyi Wang and others", title="Geometry Conflict: Explaining and Controlling Forgetting in {LLM} Continual Post-Training", howpublished="arXiv:2605.09608", year="2026")),
    "khetarpal2022": ("article", dict(author="Khimya Khetarpal and Matthew Riemer and Irina Rish and Doina Precup", title="Towards Continual Reinforcement Learning: A Review and Perspectives", journal="Journal of Artificial Intelligence Research", volume="75", pages="1401--1476", year="2022")),
    "wolczyk2021": ("inproceedings", dict(author="Maciej Wo{\\l}czyk and Micha{\\l} Zaj{\\k{a}}c and Razvan Pascanu and {\\L}ukasz Kuci{\\'n}ski and Piotr Mi{\\l}o{\\'s}", title="Continual World: A Robotic Benchmark for Continual Reinforcement Learning", booktitle="Advances in Neural Information Processing Systems (NeurIPS)", year="2021")),
    "zheng2025spurious": ("inproceedings", dict(author="Junhao Zheng and Xidi Cai and Shengjie Qiu and Qianli Ma", title="Spurious Forgetting in Continual Learning of Language Models", booktitle="International Conference on Learning Representations (ICLR)", year="2025")),
    "kotha2024implicit": ("inproceedings", dict(author="Suhas Kotha and Jacob Mitchell Springer and Aditi Raghunathan", title="Understanding Catastrophic Forgetting in Language Models via Implicit Inference", booktitle="International Conference on Learning Representations (ICLR)", year="2024")),
    "martens2010": ("inproceedings", dict(author="James Martens", title="Deep Learning via {Hessian}-Free Optimization", booktitle="Proceedings of the 27th International Conference on Machine Learning (ICML)", year="2010")),
    "martens2015kfac": ("inproceedings", dict(author="James Martens and Roger Grosse", title="Optimizing Neural Networks with {Kronecker}-Factored Approximate Curvature", booktitle="Proceedings of the 32nd International Conference on Machine Learning (ICML)", year="2015")),
    "ritter2018": ("inproceedings", dict(author="Hippolyt Ritter and Aleksandar Botev and David Barber", title="Online Structured {Laplace} Approximations for Overcoming Catastrophic Forgetting", booktitle="Advances in Neural Information Processing Systems (NeurIPS)", year="2018")),
    "lee1999nmf": ("article", dict(author="Daniel D. Lee and H. Sebastian Seung", title="Learning the Parts of Objects by Non-Negative Matrix Factorization", journal="Nature", volume="401", pages="788--791", year="1999")),
    "hinton2015": ("misc", dict(author="Geoffrey Hinton and Oriol Vinyals and Jeff Dean", title="Distilling the Knowledge in a Neural Network", howpublished="arXiv:1503.02531", year="2015")),
    "vitter1985": ("article", dict(author="Jeffrey S. Vitter", title="Random Sampling with a Reservoir", journal="ACM Transactions on Mathematical Software", volume="11", number="1", pages="37--57", year="1985")),
    "gonzalez1985": ("article", dict(author="Teofilo F. Gonzalez", title="Clustering to Minimize the Maximum Intercluster Distance", journal="Theoretical Computer Science", volume="38", pages="293--306", year="1985")),
    "altman1999": ("book", dict(author="Eitan Altman", title="Constrained {Markov} Decision Processes", publisher="Chapman and Hall/CRC", year="1999")),
    "stooke2020": ("inproceedings", dict(author="Adam Stooke and Joshua Achiam and Pieter Abbeel", title="Responsive Safety in Reinforcement Learning by {PID} {Lagrangian} Methods", booktitle="Proceedings of the 37th International Conference on Machine Learning (ICML)", year="2020")),
    "cobbe2021gsm8k": ("misc", dict(author="Karl Cobbe and others", title="Training Verifiers to Solve Math Word Problems", howpublished="arXiv:2110.14168", year="2021")),
    "clark2018arc": ("misc", dict(author="Peter Clark and others", title="Think You Have Solved Question Answering? Try {ARC}, the {AI2} Reasoning Challenge", howpublished="arXiv:1803.05457", year="2018")),
    "austin2021mbpp": ("misc", dict(author="Jacob Austin and others", title="Program Synthesis with Large Language Models", howpublished="arXiv:2108.07732", year="2021")),
    "zhou2023ifeval": ("misc", dict(author="Jeffrey Zhou and others", title="Instruction-Following Evaluation for Large Language Models", howpublished="arXiv:2311.07911", year="2023")),
    "hendrycks2021mmlu": ("inproceedings", dict(author="Dan Hendrycks and Collin Burns and Steven Basart and Andy Zou and Mantas Mazeika and Dawn Song and Jacob Steinhardt", title="Measuring Massive Multitask Language Understanding", booktitle="International Conference on Learning Representations (ICLR)", year="2021")),
    "zellers2019hellaswag": ("inproceedings", dict(author="Rowan Zellers and Ari Holtzman and Yonatan Bisk and Ali Farhadi and Yejin Choi", title="{HellaSwag}: Can a Machine Really Finish Your Sentence?", booktitle="Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (ACL)", year="2019")),
    "merity2017wikitext": ("inproceedings", dict(author="Stephen Merity and Caiming Xiong and James Bradbury and Richard Socher", title="Pointer Sentinel Mixture Models", booktitle="International Conference on Learning Representations (ICLR)", year="2017")),
    "qwen2024": ("misc", dict(author="{Qwen Team}", title="{Qwen2.5} Technical Report", howpublished="arXiv:2412.15115", year="2024")),
    "loshchilov2019": ("inproceedings", dict(author="Ilya Loshchilov and Frank Hutter", title="Decoupled Weight Decay Regularization", booktitle="International Conference on Learning Representations (ICLR)", year="2019")),
    "kingma2015": ("inproceedings", dict(author="Diederik P. Kingma and Jimmy Ba", title="Adam: A Method for Stochastic Optimization", booktitle="International Conference on Learning Representations (ICLR)", year="2015")),
    "vaswani2017": ("inproceedings", dict(author="Ashish Vaswani and others", title="Attention Is All You Need", booktitle="Advances in Neural Information Processing Systems (NeurIPS)", year="2017")),
    "brown2020": ("inproceedings", dict(author="Tom B. Brown and others", title="Language Models are Few-Shot Learners", booktitle="Advances in Neural Information Processing Systems (NeurIPS)", year="2020")),
    "henderson2018": ("inproceedings", dict(author="Peter Henderson and Riashat Islam and Philip Bachman and Joelle Pineau and Doina Precup and David Meger", title="Deep Reinforcement Learning that Matters", booktitle="Proceedings of the AAAI Conference on Artificial Intelligence", year="2018")),
    "agarwal2021": ("inproceedings", dict(author="Rishabh Agarwal and Max Schwarzer and Pablo Samuel Castro and Aaron Courville and Marc G. Bellemare", title="Deep Reinforcement Learning at the Edge of the Statistical Precipice", booktitle="Advances in Neural Information Processing Systems (NeurIPS)", year="2021")),
    "dror2018": ("inproceedings", dict(author="Rotem Dror and Gili Baumer and Segev Shlomov and Roi Reichart", title="The Hitchhiker's Guide to Testing Statistical Significance in Natural Language Processing", booktitle="Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (ACL)", year="2018")),
    "mcnemar1947": ("article", dict(author="Quinn McNemar", title="Note on the Sampling Error of the Difference Between Correlated Proportions or Percentages", journal="Psychometrika", volume="12", number="2", pages="153--157", year="1947")),
    "wilson1927": ("article", dict(author="Edwin B. Wilson", title="Probable Inference, the Law of Succession, and Statistical Inference", journal="Journal of the American Statistical Association", volume="22", number="158", pages="209--212", year="1927")),
    "efron1993": ("book", dict(author="Bradley Efron and Robert J. Tibshirani", title="An Introduction to the Bootstrap", publisher="Chapman and Hall", year="1993")),
    "benjamini1995": ("article", dict(author="Yoav Benjamini and Yosef Hochberg", title="Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing", journal="Journal of the Royal Statistical Society: Series B", volume="57", number="1", pages="289--300", year="1995")),
    "gupta2023rewarm": ("misc", dict(author="Kshitij Gupta and others", title="Continual Pre-Training of Large Language Models: How to (Re)warm Your Model?", howpublished="arXiv:2308.04014", year="2023")),
    "scialom2022": ("inproceedings", dict(author="Thomas Scialom and Tuhin Chakrabarty and Smaranda Muresan", title="Fine-tuned Language Models are Continual Learners", booktitle="Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP)", year="2022")),
    "huang2024ssr": ("inproceedings", dict(author="Jianheng Huang and others", title="Mitigating Catastrophic Forgetting in Large Language Models with Self-Synthesized Rehearsal", booktitle="Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL)", year="2024")),
    "hayou2024loraplus": ("inproceedings", dict(author="Soufiane Hayou and Nikhil Ghosh and Bin Yu", title="{LoRA+}: Efficient Low Rank Adaptation of Large Models", booktitle="Proceedings of the 41st International Conference on Machine Learning (ICML)", year="2024")),
    "liu2024dora": ("inproceedings", dict(author="Shih-Yang Liu and others", title="{DoRA}: Weight-Decomposed Low-Rank Adaptation", booktitle="Proceedings of the 41st International Conference on Machine Learning (ICML)", year="2024")),
    "zhang2023adalora": ("inproceedings", dict(author="Qingru Zhang and others", title="Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning", booktitle="International Conference on Learning Representations (ICLR)", year="2023")),
    "kalajdzievski2024": ("misc", dict(author="Damjan Kalajdzievski", title="Scaling Laws for Forgetting When Fine-Tuning Large Language Models", howpublished="arXiv:2401.05605", year="2024")),
    "ilharco2023": ("inproceedings", dict(author="Gabriel Ilharco and others", title="Editing Models with Task Arithmetic", booktitle="International Conference on Learning Representations (ICLR)", year="2023")),
    "wortsman2022": ("inproceedings", dict(author="Mitchell Wortsman and others", title="Robust Fine-Tuning of Zero-Shot Models", booktitle="Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)", year="2022")),
    "gao2020pile": ("misc", dict(author="Leo Gao and others", title="The {Pile}: An 800{GB} Dataset of Diverse Text for Language Modeling", howpublished="arXiv:2101.00027", year="2020")),
    "welbl2017sciq": ("inproceedings", dict(author="Johannes Welbl and Nelson F. Liu and Matt Gardner", title="Crowdsourcing Multiple Choice Science Questions", booktitle="Proceedings of the 3rd Workshop on Noisy User-generated Text (W-NUT)", year="2017")),
    "bisk2020piqa": ("inproceedings", dict(author="Yonatan Bisk and Rowan Zellers and Ronan Le Bras and Jianfeng Gao and Yejin Choi", title="{PIQA}: Reasoning about Physical Commonsense in Natural Language", booktitle="Proceedings of the AAAI Conference on Artificial Intelligence", year="2020")),
    "pal2022medmcqa": ("inproceedings", dict(author="Ankit Pal and Logesh Kumar Umapathi and Malaikannan Sankarasubbu", title="{MedMCQA}: A Large-scale Multi-Subject Multi-Choice Dataset for Medical Domain Question Answering", booktitle="Proceedings of the Conference on Health, Inference, and Learning (CHIL)", year="2022")),
}


def _clean(text: str) -> str:
    out = text
    for a, b in (("{\\`a}", "à"), ('{\\"u}', "ü"), ('{T\\"ulu}', "Tülu"), ("{\\l}", "ł"), ("{\\L}", "Ł"), ("{\\k{a}}", "ą"),
                 ("{\\'n}", "ń"), ("{\\'s}", "ś"), ("{\\&}", "&"), ("\\&", "&"), ("--", "–")):
        out = out.replace(a, b)
    return out.replace("{", "").replace("}", "")


def _authors_ieee(raw: str) -> str:
    names = [n.strip() for n in _clean(raw).split(" and ")]
    formatted = []
    for n in names:
        if n == "others":
            formatted.append("et al.")
            continue
        if " " not in n:
            formatted.append(n)
            continue
        parts = n.split(" ")
        last = parts[-1]
        # keep particles such as "de Masson d'Autume", "van de Weijer"
        particles = {"de", "van", "der", "von", "d'Autume", "le"}
        i = len(parts) - 1
        while i > 1 and parts[i - 1].lower() in particles | {"masson"}:
            i -= 1
        last = " ".join(parts[i:])
        initials = " ".join(p if p.endswith(".") else "-".join(x[0] + "." for x in p.split("-") if x) for p in parts[:i] if p)
        initials = initials.replace("-", "-")
        formatted.append(f"{initials} {last}".strip())
    if formatted and formatted[-1] == "et al.":
        return ", ".join(formatted[:-1]) + " et al."
    if len(formatted) <= 2:
        return " and ".join(formatted)
    return ", ".join(formatted[:-1]) + ", and " + formatted[-1]


def ieee(key: str) -> str:
    kind, f = REFS[key]
    authors = _authors_ieee(f["author"])
    title = _clean(f["title"])
    year = f["year"]
    if kind == "article":
        s = f'{authors}, "{title}," {_clean(f["journal"])}'
        if "volume" in f:
            s += f', vol. {f["volume"]}'
        if "number" in f:
            s += f', no. {f["number"]}'
        if "pages" in f:
            s += f', pp. {_clean(f["pages"])}'
        return s + f", {year}."
    if kind == "inproceedings":
        s = f'{authors}, "{title}," in {_clean(f["booktitle"])}'
        if "pages" in f:
            s += f', pp. {_clean(f["pages"])}'
        s += f", {year}"
        if "note" in f:
            s += f', {f["note"]}'
        return s + "."
    if kind == "book":
        return f"{authors}, {title}. {_clean(f['publisher'])}, {year}."
    return f'{authors}, "{title}," {f["howpublished"]}, {year}.'


def bibtex(keys: list[str]) -> str:
    chunks = []
    for key in keys:
        kind, f = REFS[key]
        body = ",\n".join(f"  {k} = {{{v}}}" for k, v in f.items())
        chunks.append(f"@{kind}{{{key},\n{body}\n}}")
    return "\n\n".join(chunks) + "\n"
