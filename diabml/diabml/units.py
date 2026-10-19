"""Physical constants and unit conversions used across diabml."""

HARTREE_EV = 27.211386245988      # eV per hartree
EV_KCAL = 23.060548               # kcal/mol per eV
HARTREE_KCAL = HARTREE_EV * EV_KCAL
BOHR_ANG = 0.529177210903         # Angstrom per bohr
COULOMB_EV_ANG = 14.399645        # e^2 / (4 pi eps0) in eV * Angstrom


def ev_to_kcal(x):
    return x * EV_KCAL


def kcal_to_ev(x):
    return x / EV_KCAL
