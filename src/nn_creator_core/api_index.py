from solver.master.ai_objects.AE1D import AE1D
from solver.master.ai_objects.FNN1D import FNN1D
from solver.master.models.custom_layers import Sampling, DenseBlock_small, OneHotDecoder
from solver.master.preparators.curve_fitting import BezierTransformer
from solver.master.preparators.deltas_preparer import DeltaPreparer
from solver.master.preparators.formulas_preparer import FormulaPreparer
from solver.master.utils.uncertainity import checker

MODEL_TYPES = {"FNN1D": FNN1D, "AE1D": AE1D}

PREPARERS = {"delta": DeltaPreparer, "bezier_curve": BezierTransformer, "formulas": FormulaPreparer}

CHECKERS = {"speedlines": checker}

