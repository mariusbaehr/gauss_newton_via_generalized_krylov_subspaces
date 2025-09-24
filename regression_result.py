import numpy.typing as npt

class RegressionResult:
    """
    Representation, of the regression results

    Attributes
    ----------

    method_name: Name of the method used.
    x: Solution for the regression parameters
    success: 
    nrev: 
    full_njev:
    njev: !!!!!!!FAlls nur jacobi mal vektor!!!
    nit:

    """

    method_name: str
    x: npt.NDArray
    success: bool
    nrev: int | None
    full_njev: int | None
    njev: int | None
    nit: int


    def __init__(self, method_name: str, x: npt.NDArray , success: bool, nrev: int | None, full_njev: int | None, njev: int | None, nit: int):
        self.method_name = method_name
        self.x = x
        self.success = success
        self.nrev = nrev
        self.full_njev = full_njev
        self.njev = njev
        self.nit = nit

    def __str__(self):
        success_message = "converged successfuly to" if self.success else "failed to terminate and stopped at"
        return f"{self.method_name} {success_message} {self.x}. After {self.nit} iterations using {self.nrev} evaluations of the residual, "



