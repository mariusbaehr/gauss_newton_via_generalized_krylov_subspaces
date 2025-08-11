import numpy.typing as npt

class RegressionResult:
    """
    Representation, of the regression results

    Attributes
    ----------

    method_name: Name of the method used.
    beta: Solution for the regression parameters
    success: 
    nrev: 
    njev:
    nj_times_v_ev: !!!!!!!FAlls nur jacobi mal vektor!!!
    nit:

    """

    method_name: str
    beta: npt.NDArray
    success: bool
    nrev: int | None
    njev: int | None
    nit: int


    def __init__(self, method_name: str, beta: npt.NDArray , success: bool, nrev: int | None, njev: int | None, nit: int):
        self.method_name = method_name
        self.beta = beta
        self.success = success
        self.nrev = nrev
        self.njev = njev
        self.nit = nit

    def __str__(self):
        success_message = "converged successfuly to" if self.success else "failed to terminate and stopped at"
        return f"{self.method_name} {success_message} {self.beta}. After {self.nit} iterations using {self.nrev} evaluations of the residual, "



