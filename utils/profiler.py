import contextlib
import time 
import torch

class Profile(contextlib.ContextDecorator):
    """
    Profile class for profiling execution time. 
    Can be used as a decorator with @Profile() or as a context manager with 'with Profile():'.

    Attributes
    ----------
    t : float
        Accumulated time.
    cuda : bool
        Indicates whether CUDA is available.

    Methods
    -------
    __enter__()
        Starts timing.
    __exit__(type, value, traceback)
        Stops timing and updates accumulated time.
    time()
        Returns the current time, synchronizing with CUDA if available.
    """
    
    def __init__(self, t=0.0):
        """
        Initializes the Profile class.

        Parameters:
        t : float
            Initial accumulated time. Defaults to 0.0.
        """
        self.t = t  # Accumulated time
        self.cuda = torch.cuda.is_available()  # Checks if CUDA is available

    def __enter__(self):
        """
        Starts timing.
        
        Returns:
        self
        """
        self.start = self.time()  # Start time

        return self

    def __exit__(self, type, value, traceback):
        """
        Stops timing and updates accumulated time.
        
        Parameters:
        type, value, traceback : 
            Standard parameters for an exit method in a context manager.
        """
        self.dt = self.time() - self.start  # Delta-time
        self.t += self.dt  # Accumulates delta-time

    def time(self):
        """
        Returns the current time, synchronizing with CUDA if available.
        
        Returns:
        float
            The current time.
        """
        if self.cuda:  # If CUDA is available
            torch.cuda.synchronize()  # Synchronizes with CUDA
        return time.time()  # Returns current time