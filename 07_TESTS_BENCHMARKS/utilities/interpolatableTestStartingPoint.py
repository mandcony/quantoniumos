# -*- coding: utf-8 -*-
#
# QuantoniumOS Test Suite
# Testing with QuantoniumOS implementations
#
# ===================================================================

import unittest
import sys
import os
import numpy as np
from binascii import unhexlify

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Import QuantoniumOS RFT algorithms
try:
    sys.path.insert(0, '/workspaces/quantoniumos/04_RFT_ALGORITHMS')
    from canonical_true_rft import CanonicalTrueRFT
    from true_rft_exact import TrueRFTExact
    from true_rft_engine_bindings import TrueRFTEngineBindings
except ImportError as e:
    print(f"Warning: Could not import RFT algorithms: {e}")

# Import QuantoniumOS quantum engines
try:
    sys.path.insert(0, '/workspaces/quantoniumos/05_QUANTUM_ENGINES')
    from bulletproof_quantum_kernel import BulletproofQuantumKernel
    from topological_quantum_kernel import TopologicalQuantumKernel
    from topological_vertex_engine import TopologicalVertexEngine
    from topological_vertex_geometric_engine import TopologicalVertexGeometricEngine
    from vertex_engine_canonical import VertexEngineCanonical
    from working_quantum_kernel import WorkingQuantumKernel
    from true_rft_engine_bindings import TrueRFTEngineBindings as QuantumRFTBindings
except ImportError as e:
    print(f"Warning: Could not import quantum engines: {e}")

# Import QuantoniumOS cryptography modules
try:
    sys.path.insert(0, '/workspaces/quantoniumos/06_CRYPTOGRAPHY')
    from quantonium_crypto_production import QuantoniumCrypto
    from true_rft_feistel_bindings import TrueRFTFeistel
except ImportError as e:
    print(f"Warning: Could not import cryptography modules: {e}")

# Import QuantoniumOS validators
try:
    sys.path.insert(0, '/workspaces/quantoniumos/02_CORE_VALIDATORS')
    from basic_scientific_validator import BasicScientificValidator
    from definitive_quantum_validation import DefinitiveQuantumValidation
    from phd_level_scientific_validator import PhdLevelScientificValidator
    from publication_ready_validation import PublicationReadyValidation
except ImportError as e:
    print(f"Warning: Could not import validators: {e}")

# Import QuantoniumOS running systems
try:
    sys.path.insert(0, '/workspaces/quantoniumos/03_RUNNING_SYSTEMS')
    from app import app
    from main import main
    from quantonium import QuantoniumOS
except ImportError as e:
    print(f"Warning: Could not import running systems: {e}")

from .interpolatableHelpers import *

def test_starting_point(glyph0, glyph1, ix, tolerance, matching):
    if matching is None:
        matching = list(range(len(glyph0.isomorphisms)))
    contour0 = glyph0.isomorphisms[ix]
    contour1 = glyph1.isomorphisms[matching[ix]]
    m0Vectors = glyph0.greenVectors
    m1Vectors = [glyph1.greenVectors[i] for i in matching]

    c0 = contour0[0]
    # Next few lines duplicated below.
    costs = [vdiff_hypot2_complex(c0[0], c1[0]) for c1 in contour1]
    min_cost_idx, min_cost = min(enumerate(costs), key=lambda x: x[1])
    first_cost = costs[0]
    proposed_point = contour1[min_cost_idx][1]
    reverse = contour1[min_cost_idx][2]

    if min_cost < first_cost * tolerance:
        # c0 is the first isomorphism of the m0 master
        # contour1 is list of all isomorphisms of the m1 master
        #
        # If the two shapes are both circle-ish and slightly
        # rotated, we detect wrong start point. This is for
        # example the case hundreds of times in
        # RobotoSerif-Italic[GRAD,opsz,wdth,wght].ttf
        #
        # If the proposed point is only one off from the first
        # point (and not reversed), try harder:
        #
        # Find the major eigenvector of the covariance matrix,
        # and rotate the contours by that angle. Then find the
        # closest point again.  If it matches this time, let it
        # pass.

        num_points = len(glyph1.points[ix])
        leeway = 3
        if not reverse and (
            proposed_point <= leeway or proposed_point >= num_points - leeway
        ):
            # Try harder

            # Recover the covariance matrix from the GreenVectors.
            # This is a 2x2 matrix.
            transforms = []
            for vector in (m0Vectors[ix], m1Vectors[ix]):
                vector[1]
                vector[2]
                stddevX = vector[3] * 0.5
                stddevY = vector[4] * 0.5
                correlation = vector[5]
                if correlation:
                    correlation /= abs(vector[0])

                # https://cookierobotics.com/007/
                a = stddevX * stddevX  # VarianceX
                c = stddevY * stddevY  # VarianceY
                b = correlation * stddevX * stddevY  # Covariance

                delta = (((a - c) * 0.5) ** 2 + b * b) ** 0.5
                lambda1 = (a + c) * 0.5 + delta  # Major eigenvalue
                lambda2 = (a + c) * 0.5 - delta  # Minor eigenvalue
                theta = atan2(lambda1 - a, b) if b != 0 else (pi * 0.5 if a < c else 0)
                trans = Transform()
                # Don't translate here. We are working on the complex-vector
                # that includes more than just the points. It's horrible what
                # we are doing anyway...
                # trans = trans.translate(meanX, meanY)
                trans = trans.rotate(theta)
                trans = trans.scale(sqrt(lambda1), sqrt(lambda2))
                transforms.append(trans)

            trans = transforms[0]
            new_c0 = (
                [complex(*trans.transformPoint((pt.real, pt.imag))) for pt in c0[0]],
            ) + c0[1:]
            trans = transforms[1]
            new_contour1 = []
            for c1 in contour1:
                new_c1 = (
                    [
                        complex(*trans.transformPoint((pt.real, pt.imag)))
                        for pt in c1[0]
                    ],
                ) + c1[1:]
                new_contour1.append(new_c1)

            # Next few lines duplicate from above.
            costs = [
                vdiff_hypot2_complex(new_c0[0], new_c1[0]) for new_c1 in new_contour1
            ]
            min_cost_idx, min_cost = min(enumerate(costs), key=lambda x: x[1])
            first_cost = costs[0]
            if min_cost < first_cost * tolerance:
                # Don't report this
                # min_cost = first_cost
                # reverse = False
                # proposed_point = 0  # new_contour1[min_cost_idx][1]
                pass

    this_tolerance = min_cost / first_cost if first_cost else 1
    log.debug(
        "test-starting-point: tolerance %g",
        this_tolerance,
    )
    return this_tolerance, proposed_point, reverse
