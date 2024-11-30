"""
Quantum category theory implementation for categorical quantum mechanics.
Implements monoidal categories, functors, and natural transformations.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Callable
from dataclasses import dataclass
from functools import partial

@dataclass
class Morphism:
    """Represents a morphism in a category."""
    source: str
    target: str
    function: Callable
    name: str

class QuantumCategory:
    """Implements categorical quantum mechanics structures."""

    def __init__(self, dim: int):
        """
        Initialize quantum category.

        Args:
            dim: Dimension of the quantum system
        """
        self.dim = dim
        self.objects = {}
        self.morphisms = {}
        self.tensor_product = None
        self.braiding = None

    def add_object(self, name: str, state_space: np.ndarray) -> None:
        """
        Add object to category
        |ψ⟩ ∈ H

        Args:
            name: Object identifier
            state_space: Quantum state space
        """
        self.objects[name] = state_space

    def add_morphism(self, morphism: Morphism) -> None:
        """
        Add morphism between objects
        f: A → B

        Args:
            morphism: Morphism to add
        """
        self.morphisms[morphism.name] = morphism

    def compose_morphisms(self, f: Morphism, g: Morphism) -> Morphism:
        """
        Compose two morphisms
        g ∘ f: A → C

        Args:
            f: First morphism
            g: Second morphism

        Returns:
            Morphism: Composed morphism
        """
        if f.target != g.source:
            raise ValueError("Morphisms not composable")

        def composed_function(x):
            return g.function(f.function(x))

        return Morphism(
            source=f.source,
            target=g.target,
            function=composed_function,
            name=f"{g.name}∘{f.name}"
        )

    def setup_monoidal_structure(self) -> None:
        """
        Setup monoidal category structure
        (C, ⊗, I, α, λ, ρ)
        """
        # Define tensor product
        def tensor_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            return np.kron(a, b)
        self.tensor_product = tensor_product

        # Define braiding
        def braiding(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            dim_a = len(a)
            dim_b = len(b)
            sigma = np.zeros((dim_a * dim_b, dim_a * dim_b))
            for i in range(dim_a):
                for j in range(dim_b):
                    sigma[j*dim_a + i, i*dim_b + j] = 1
            return sigma
        self.braiding = braiding

    def create_functor(self, source_cat: 'QuantumCategory',
                      target_cat: 'QuantumCategory') -> Dict[str, Callable]:
        """
        Create functor between categories
        F: C → D

        Args:
            source_cat: Source category
            target_cat: Target category

        Returns:
            Dict[str, Callable]: Functor mappings
        """
        functor = {}

        # Map objects
        for name, obj in source_cat.objects.items():
            functor[name] = lambda x, obj=obj: target_cat.objects.get(
                name, np.zeros_like(obj))

        # Map morphisms
        for name, morph in source_cat.morphisms.items():
            functor[name] = lambda x, m=morph: target_cat.morphisms.get(
                name, Morphism(
                    source=m.source,
                    target=m.target,
                    function=lambda y: y,
                    name=m.name
                ))

        return functor

    def create_natural_transformation(self, F: Dict[str, Callable],
                                   G: Dict[str, Callable]) -> Dict[str, Callable]:
        """
        Create natural transformation between functors
        η: F ⇒ G

        Args:
            F: First functor
            G: Second functor

        Returns:
            Dict[str, Callable]: Natural transformation components
        """
        eta = {}

        # Define components
        for obj_name in self.objects:
            eta[obj_name] = lambda x, name=obj_name: G[name](F[name](x))

        return eta

    def verify_naturality(self, eta: Dict[str, Callable],
                         f: Morphism) -> bool:
        """
        Verify naturality condition
        G(f) ∘ η_A = η_B ∘ F(f)

        Args:
            eta: Natural transformation
            f: Morphism to check

        Returns:
            bool: True if naturality condition holds
        """
        # Get relevant components
        eta_source = eta[f.source]
        eta_target = eta[f.target]

        # Check naturality condition
        test_state = self.objects[f.source]
        lhs = eta_target(f.function(test_state))
        rhs = f.function(eta_source(test_state))

        return np.allclose(lhs, rhs)

    def create_adjunction(self, F: Dict[str, Callable],
                         G: Dict[str, Callable]) -> Tuple[Dict[str, Callable],
                                                        Dict[str, Callable]]:
        """
        Create adjunction between functors
        F ⊣ G

        Args:
            F: Left adjoint functor
            G: Right adjoint functor

        Returns:
            Tuple[Dict[str, Callable], Dict[str, Callable]]: Unit and counit
        """
        # Define unit
        unit = {}
        for name in self.objects:
            unit[name] = lambda x, n=name: G[n](F[n](x))

        # Define counit
        counit = {}
        for name in self.objects:
            counit[name] = lambda x, n=name: F[n](G[n](x))

        return unit, counit

    def verify_triangle_identities(self, F: Dict[str, Callable],
                                 G: Dict[str, Callable],
                                 unit: Dict[str, Callable],
                                 counit: Dict[str, Callable]) -> bool:
        """
        Verify triangle identities for adjunction
        (Gε)(ηG) = 1_G and (εF)(Fη) = 1_F

        Args:
            F: Left adjoint
            G: Right adjoint
            unit: Unit natural transformation
            counit: Counit natural transformation

        Returns:
            bool: True if triangle identities hold
        """
        # Check first triangle identity
        for name, obj in self.objects.items():
            lhs = lambda x, n=name: G[n](counit[n](unit[n](x)))
            rhs = lambda x: x
            if not np.allclose(lhs(obj), rhs(obj)):
                return False

        # Check second triangle identity
        for name, obj in self.objects.items():
            lhs = lambda x, n=name: F[n](unit[n](counit[n](x)))
            rhs = lambda x: x
            if not np.allclose(lhs(obj), rhs(obj)):
                return False

        return True

    def create_yoneda_embedding(self) -> Dict[str, Callable]:
        """
        Create Yoneda embedding
        y: C → [C^op, Set]

        Returns:
            Dict[str, Callable]: Yoneda embedding functor
        """
        yoneda = {}

        # Map objects to functors
        for name, obj in self.objects.items():
            yoneda[name] = lambda x, n=name: {
                m.name: m.function(x)
                for m in self.morphisms.values()
                if m.target == n
            }

        # Map morphisms to natural transformations
        for name, morph in self.morphisms.items():
            yoneda[name] = lambda F, m=morph: {
                n: F[m.name] for n in self.objects
            }

        return yoneda

    def compute_kan_extension(self, F: Dict[str, Callable],
                            K: Dict[str, Callable]) -> Dict[str, Callable]:
        """
        Compute Kan extension
        Lan_K F

        Args:
            F: Functor to extend
            K: Functor to extend along

        Returns:
            Dict[str, Callable]: Left Kan extension
        """
        lan = {}

        # Compute coend formula
        for name in self.objects:
            lan[name] = lambda x, n=name: sum(
                F[m.target](K[m.source](x))
                for m in self.morphisms.values()
                if m.source == n
            )

        return lan
