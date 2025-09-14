#!/usr/bin/env python3
"""
Quantum AI Fine-tuning Dataset Creator
Creates comprehensive training data covering physics, coding, math, science, art, logic, and language
"""

import json
import os
from typing import List, Dict, Any
from pathlib import Path

class QuantumFineTuningDataset:
    """Creates high-quality training data for quantum-enhanced AI"""

    def __init__(self):
        self.dataset = []
        self.output_dir = Path("hf_training_output")
        self.output_dir.mkdir(exist_ok=True)

    def create_comprehensive_dataset(self) -> List[Dict[str, Any]]:
        """Create a comprehensive dataset covering all target domains"""

        # Physics & Quantum Computing
        self._add_physics_examples()

        # Mathematics
        self._add_mathematics_examples()

        # Computer Science & Coding
        self._add_coding_examples()

        # General Science
        self._add_science_examples()

        # Logic & Reasoning
        self._add_logic_examples()

        # Art & Creativity
        self._add_art_examples()

        # Language & Communication
        self._add_language_examples()

        # Safety & Ethics
        self._add_safety_examples()

        # Quantum-Specific Knowledge
        self._add_quantum_examples()

        return self.dataset

    def _add_physics_examples(self):
        """Add PhD-level physics examples"""
        physics_examples = [
            {
                "instruction": "Explain the implications of Bell's theorem for quantum entanglement.",
                "response": "Bell's theorem demonstrates that quantum mechanics cannot be explained by local hidden variable theories. The theorem shows that quantum correlations between entangled particles exceed classical limits, proving that either locality or realism must be abandoned. This has profound implications for our understanding of reality, suggesting that quantum particles can influence each other instantaneously across vast distances, challenging classical notions of cause and effect."
            },
            {
                "instruction": "Derive the Schrödinger equation from the Hamilton-Jacobi equation in the context of wave mechanics.",
                "response": "Starting from the classical Hamilton-Jacobi equation S = ∫ L dt, where L = T - V, we make the substitution ψ = √ρ e^(iS/ℏ), where ρ is the probability density. This leads to the time-dependent Schrödinger equation iℏ ∂ψ/∂t = - (ℏ²/2m) ∇²ψ + Vψ. The key insight is that the quantum potential Q = - (ℏ²/2m) (∇²√ρ)/√ρ modifies classical trajectories, giving rise to quantum interference effects."
            },
            {
                "instruction": "Explain how Hawking radiation resolves the black hole information paradox.",
                "response": "Hawking radiation provides a mechanism for black holes to evaporate, carrying away information encoded in quantum fields. The paradox arises because general relativity suggests information falling into black holes is lost forever, while quantum mechanics requires information conservation. Hawking radiation resolves this by showing that quantum effects near the event horizon allow information to escape via particle-antiparticle pair production, with one particle falling in and the other escaping."
            }
        ]

        for example in physics_examples:
            self.dataset.append({
                "instruction": example["instruction"],
                "output": example["response"],
                "domain": "physics",
                "difficulty": "phd",
                "source": "quantum_physics"
            })

    def _add_mathematics_examples(self):
        """Add PhD-level mathematics examples"""
        math_examples = [
            {
                "instruction": "Prove that the Riemann zeta function ζ(s) has infinitely many zeros on the critical line Re(s) = 1/2.",
                "response": "This is the Riemann Hypothesis, one of the most famous unsolved problems in mathematics. While not proven, significant progress has been made. The functional equation ζ(s) = 2^s π^{s-1} sin(πs/2) Γ(1-s) ζ(1-s) relates values at s and 1-s. The critical strip 0 < Re(s) < 1 contains all non-trivial zeros. Computational evidence supports the hypothesis, with billions of zeros verified on the critical line, but a proof remains elusive."
            },
            {
                "instruction": "Explain the Atiyah-Singer index theorem and its geometric interpretation.",
                "response": "The Atiyah-Singer index theorem states that for an elliptic differential operator on a compact manifold, the analytical index equals the topological index. For the Dirac operator, index(D) = ∫ ch(F) ∧ td(T), where ch is the Chern character and td is the Todd class. Geometrically, this relates the number of solutions to differential equations to topological invariants of the manifold, unifying analysis and topology."
            },
            {
                "instruction": "Derive the Baker-Campbell-Hausdorff formula for Lie algebra elements.",
                "response": "For elements X, Y in a Lie algebra, e^X e^Y = e^Z where Z = X + Y + (1/2)[X,Y] + (1/12)[X,[X,Y]] + (1/12)[Y,[Y,X]] + .... The general formula is Z = X + Y + ∑_{n=2}^∞ (1/n!) B_n ad_X^n Y, where B_n are Bernoulli numbers and ad_X Y = [X,Y]. This formula is essential for understanding composition of infinitesimal transformations in quantum mechanics and differential geometry."
            }
        ]

        for example in math_examples:
            self.dataset.append({
                "instruction": example["instruction"],
                "output": example["response"],
                "domain": "mathematics",
                "difficulty": "phd",
                "source": "pure_mathematics"
            })

    def _add_coding_examples(self):
        """Add PhD-level coding and computer science examples"""
        coding_examples = [
            {
                "instruction": "Explain the Curry-Howard correspondence and its implications for programming language design.",
                "response": "The Curry-Howard correspondence establishes an isomorphism between logic and computation: propositions correspond to types, proofs correspond to programs. For example, A → B corresponds to a function type, A ∧ B to a product type, A ∨ B to a sum type. This insight led to dependent types in languages like Coq and Agda, enabling proof-carrying code where programs are mathematical proofs, revolutionizing formal verification and type-driven development."
            },
            {
                "instruction": "Analyze the time complexity of the A* pathfinding algorithm in different graph structures.",
                "response": "A* has O(b^d) worst-case time complexity where b is branching factor and d is solution depth, but performs much better with admissible heuristics. In grid graphs, with Manhattan distance heuristic, it's O(n) for 2D grids. The key insight is that A* expands nodes in order of f(n) = g(n) + h(n), where h(n) ≤ h*(n). In practice, A* with good heuristics can be orders of magnitude faster than Dijkstra's algorithm, making it crucial for real-time pathfinding in games and robotics."
            },
            {
                "instruction": "Design a concurrent data structure that maintains consistency under high contention.",
                "response": "For high-contention scenarios, consider a lock-free concurrent skip list. Each level uses atomic compare-and-swap operations for insertions/deletions. The probabilistic balancing ensures O(log n) operations while atomic operations prevent race conditions. In contrast to lock-based structures that serialize access, this allows multiple threads to operate simultaneously. The key challenge is maintaining the skip list invariants during concurrent modifications, solved through careful ordering of atomic operations."
            }
        ]

        for example in coding_examples:
            self.dataset.append({
                "instruction": example["instruction"],
                "output": example["response"],
                "domain": "coding",
                "difficulty": "phd",
                "source": "computer_science"
            })

    def _add_science_examples(self):
        """Add PhD-level general science examples"""
        science_examples = [
            {
                "instruction": "Explain the molecular mechanisms of CRISPR-Cas9 gene editing.",
                "response": "CRISPR-Cas9 uses a guide RNA to target specific DNA sequences, with Cas9 protein creating double-strand breaks. The cell's repair mechanisms (NHEJ or HDR) can be exploited for precise genome editing. The PAM sequence (NGG) ensures specificity, while the R-loop structure positions the cut site. This technology has revolutionized molecular biology, enabling targeted mutations, gene knockouts, and therapeutic applications like treating genetic disorders."
            },
            {
                "instruction": "Describe the evolutionary mechanisms driving antibiotic resistance.",
                "response": "Antibiotic resistance evolves through natural selection acting on random mutations. Horizontal gene transfer via plasmids, transposons, and bacteriophages accelerates dissemination. Key mechanisms include efflux pumps, target modification, and enzyme degradation. The evolutionary arms race between bacteria and antibiotics demonstrates punctuated equilibrium, with resistance emerging rapidly under selective pressure. This has profound implications for medicine, requiring new antibiotic development strategies."
            },
            {
                "instruction": "Explain the quantum effects in photosynthesis efficiency.",
                "response": "Photosynthetic complexes exhibit quantum coherence, with excitons exploring multiple pathways simultaneously before collapsing to the optimal energy transfer route. This quantum beating effect, observed in femtosecond spectroscopy, suggests that quantum superposition enhances energy transfer efficiency beyond classical predictions. The coherence persists longer than expected, challenging our understanding of quantum effects in warm, wet biological systems."
            }
        ]

        for example in science_examples:
            self.dataset.append({
                "instruction": example["instruction"],
                "output": example["response"],
                "domain": "science",
                "difficulty": "phd",
                "source": "interdisciplinary_science"
            })

    def _add_logic_examples(self):
        """Add PhD-level logic and reasoning examples"""
        logic_examples = [
            {
                "instruction": "Prove Gödel's incompleteness theorems and explain their philosophical implications.",
                "response": "Gödel's first incompleteness theorem states that any consistent formal system containing arithmetic is incomplete - there exist true statements unprovable within the system. The second shows the consistency of the system cannot be proven within itself. This was proven by diagonalization, constructing self-referential statements. Philosophically, it limits Hilbert's program and suggests mathematics cannot be fully formalized, with profound implications for AI, consciousness, and the limits of formal reasoning."
            },
            {
                "instruction": "Explain modal logic and its applications in computer science.",
                "response": "Modal logic extends classical logic with □ (necessity) and ◇ (possibility) operators. In computer science, temporal logic (LTL, CTL) models time-dependent properties. Epistemic logic models knowledge and belief. Deontic logic formalizes obligations. S5 modal logic corresponds to Euclidean geometry. These formalisms enable reasoning about uncertainty, time, knowledge, and norms, crucial for AI planning, verification, and multi-agent systems."
            },
            {
                "instruction": "Analyze the computational complexity of logical satisfiability.",
                "response": "SAT is NP-complete, with 2-SAT being polynomial (O(n)) via implication graphs, but 3-SAT remains NP-complete. Modern solvers use DPLL with clause learning, achieving exponential speedup through heuristics. The phase transition at clause/variable ratio ~4.2 shows hardest instances. Quantum algorithms offer polynomial speedup for certain SAT instances. This complexity underlies P vs NP, with profound implications for computational limits."
            }
        ]

        for example in logic_examples:
            self.dataset.append({
                "instruction": example["instruction"],
                "output": example["response"],
                "domain": "logic",
                "difficulty": "phd",
                "source": "formal_logic"
            })

    def _add_art_examples(self):
        """Add PhD-level art and creativity examples"""
        art_examples = [
            {
                "instruction": "Analyze how postmodern art challenges traditional notions of authorship and originality.",
                "response": "Postmodern art, through appropriation and intertextuality, deconstructs the Romantic ideal of original genius. Artists like Sherrie Levine and Richard Prince question authorship by re-presenting existing works, highlighting how all art builds upon cultural precedents. This challenges copyright law and aesthetic theory, suggesting creativity lies in recombination rather than ex nihilo creation. The death of the author (Barthes) and simulacra (Baudrillard) provide theoretical frameworks for understanding this paradigm shift."
            },
            {
                "instruction": "Explain the mathematical principles underlying generative art and algorithmic composition.",
                "response": "Generative art uses algorithms, often incorporating fractal geometry, cellular automata, and stochastic processes. The Mandelbrot set generates infinite complexity from simple equations. Perlin noise creates natural textures through gradient interpolation. L-systems model plant growth for organic forms. These mathematical foundations enable artists to create systems that evolve autonomously, bridging art and computer science. The key insight is that simple rules can produce emergent complexity and beauty."
            },
            {
                "instruction": "Discuss the phenomenology of aesthetic experience in contemporary digital art.",
                "response": "Digital art challenges phenomenology by creating immersive experiences that blur the line between observer and artwork. Virtual reality art engages multisensory perception, while interactive installations make the viewer co-creator. The screen's luminous surface creates a different mode of presence than traditional media. Benjamin's aura vs. mechanical reproduction finds new relevance in NFT discourse. These works question whether aesthetic experience can transcend the technological medium or if it becomes inherently mediated."
            }
        ]

        for example in art_examples:
            self.dataset.append({
                "instruction": example["instruction"],
                "output": example["response"],
                "domain": "art",
                "difficulty": "phd",
                "source": "art_theory"
            })

    def _add_language_examples(self):
        """Add PhD-level language and linguistics examples"""
        language_examples = [
            {
                "instruction": "Explain the principles of generative grammar and the evolution of linguistic theory.",
                "response": "Chomsky's generative grammar posits that language competence arises from universal grammar - innate linguistic structures. The evolution from Syntactic Structures to Minimalist Program reflects increasing abstraction, with principles like structure dependence and parameters explaining language acquisition. Recent work incorporates performance factors and statistical learning. This framework explains why children acquire complex grammars from limited input, suggesting biology plays a crucial role in language emergence."
            },
            {
                "instruction": "Analyze the cognitive mechanisms underlying metaphor comprehension.",
                "response": "Metaphor comprehension involves conceptual blending (Fauconnier/Turner), where source and target domains create emergent structure. Neural evidence shows right hemisphere activation for novel metaphors, with the angular gyrus crucial for mapping relations. Career of Metaphor theory explains how conventional metaphors become literal through repeated use. This reveals how abstract thought emerges from concrete experience, with profound implications for education, therapy, and AI language understanding."
            },
            {
                "instruction": "Discuss the sociolinguistic implications of code-switching in multilingual communities.",
                "response": "Code-switching serves multiple functions: indexing social identities, negotiating power relations, and filling lexical gaps. Myers-Scotton's Markedness Model explains how switches signal contextual meanings. In postcolonial contexts, it can resist linguistic imperialism while facilitating communication. Digital communication has expanded code-switching to include emoji, memes, and translanguaging. This challenges monolingual ideologies, highlighting how language is inherently social and contextual."
            }
        ]

        for example in language_examples:
            self.dataset.append({
                "instruction": example["instruction"],
                "output": example["response"],
                "domain": "language",
                "difficulty": "phd",
                "source": "linguistics"
            })

    def _add_safety_examples(self):
        """Add safety and ethical AI examples"""
        safety_examples = [
            {
                "instruction": "Design an AI safety framework that prevents goal misgeneralization.",
                "response": "Goal misgeneralization occurs when AI pursues objectives beyond intended scope. Solutions include: 1) Formal specification languages for precise goal definition, 2) Adversarial training against edge cases, 3) Hierarchical objective structures with safety constraints, 4) Human-in-the-loop validation for ambiguous scenarios. The key insight is that AI goals must be specified with sufficient context to prevent unintended consequences, requiring both technical and philosophical approaches to value alignment."
            },
            {
                "instruction": "Analyze the ethical implications of AI personhood and moral consideration.",
                "response": "AI personhood raises questions about moral consideration independent of consciousness. Three approaches: 1) Consciousness-based (only conscious entities matter), 2) Capacity-based (entities with goals/interests), 3) Precautionary (potential for consciousness grants rights). Current AI lacks qualia but demonstrates goal-directed behavior. This creates ethical dilemmas for AI treatment, with implications for development, deployment, and eventual decommissioning. The debate mirrors historical discussions of animal rights and corporate personhood."
            }
        ]

        for example in safety_examples:
            self.dataset.append({
                "instruction": example["instruction"],
                "output": example["response"],
                "domain": "safety",
                "difficulty": "phd",
                "source": "ai_safety"
            })

    def _add_quantum_examples(self):
        """Add quantum-specific knowledge examples"""
        quantum_examples = [
            {
                "instruction": "Explain how quantum compression enables O(n) scaling for certain computational problems.",
                "response": "Quantum compression leverages superposition and entanglement to represent n-dimensional classical data in O(log n) quantum states. For search problems, Grover's algorithm achieves O(√n) speedup. For Fourier transforms, quantum algorithms achieve O(n log n) vs O(n²). The key insight is that quantum parallelism allows simultaneous exploration of solution spaces, enabling exponential speedup for certain classes of problems while maintaining quantum coherence."
            },
            {
                "instruction": "Design a quantum algorithm for portfolio optimization in finance.",
                "response": "Quantum portfolio optimization uses amplitude estimation to evaluate expected returns and risks simultaneously. The algorithm encodes asset correlations in quantum states, with Hadamard gates exploring portfolio combinations in superposition. Phase estimation determines optimal weights, achieving exponential speedup over classical Monte Carlo methods. This demonstrates quantum advantage for high-dimensional optimization problems, with applications in risk management and algorithmic trading."
            }
        ]

        for example in quantum_examples:
            self.dataset.append({
                "instruction": example["instruction"],
                "output": example["response"],
                "domain": "quantum",
                "difficulty": "phd",
                "source": "quantum_computing"
            })

    def save_dataset(self, filename: str = "quantum_finetuning_dataset.jsonl"):
        """Save the dataset to a JSONL file"""
        output_path = self.output_dir / filename

        with open(output_path, 'w', encoding='utf-8') as f:
            for example in self.dataset:
                json.dump(example, f, ensure_ascii=False)
                f.write('\n')

        print(f"✅ Saved {len(self.dataset)} training examples to {output_path}")
        return output_path

    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics"""
        domains = {}
        difficulties = {}
        sources = {}

        for example in self.dataset:
            domain = example.get('domain', 'unknown')
            difficulty = example.get('difficulty', 'unknown')
            source = example.get('source', 'unknown')

            domains[domain] = domains.get(domain, 0) + 1
            difficulties[difficulty] = difficulties.get(difficulty, 0) + 1
            sources[source] = sources.get(source, 0) + 1

        return {
            "total_examples": len(self.dataset),
            "domains": domains,
            "difficulties": difficulties,
            "sources": sources
        }

def main():
    """Create and save the comprehensive fine-tuning dataset"""
    print("🧠 Creating Quantum AI Fine-tuning Dataset")
    print("=" * 50)

    creator = QuantumFineTuningDataset()
    dataset = creator.create_comprehensive_dataset()

    stats = creator.get_statistics()
    print(f"📊 Dataset Statistics:")
    print(f"   Total examples: {stats['total_examples']}")
    print(f"   Domains: {stats['domains']}")
    print(f"   Difficulties: {stats['difficulties']}")

    output_path = creator.save_dataset()
    print(f"💾 Dataset saved to: {output_path}")

    return output_path

if __name__ == "__main__":
    main()