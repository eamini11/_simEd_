import numpy.typing

# use numpy.random's MT19937 for random-number generation w/ streams
from numpy.random import MT19937, Generator, SeedSequence

######################################################################
class rng:
    ''' This class implements a wrapper around numpy's MT19937 generator
        to allow for a streams implementation, i.e., where we can have a
        different stream of random numbers for each different stochastic
        component.  

        Each wrapper method will do the right thing to pull and then update 
        the state of the particular stream.
    '''

    # class-level variables
    _num_streams: int = 128  # arbitrary
    _streams:     list[numpy.random.Generator] = [None] * _num_streams

    ############################################################################
    @classmethod
    def numStreams(cls) -> int: return cls._num_streams

    ############################################################################
    @classmethod
    def set_seed(cls, seed: int | None = None) -> None:
        # https://numpy.org/doc/stable/reference/random/bit_generators/mt19937.html
        seed_seq = SeedSequence(seed)
        bit_generator = MT19937(seed_seq)
        
        for i in range(cls._num_streams):
            cls._streams[i] = Generator(bit_generator)
            # chain the BitGenerators
            bit_generator = bit_generator.jumped()
        
    ############################################################################
    @classmethod
    def uniform(cls, a: float, b: float, which_stream: int = 0) -> numpy.float64:
        ''' class-level method to generate floating-point values uniformly
            in [a,b), or in (a,b) if exclude_a is True
        Parameters:
            a: floating-point minimum value of the distribution
            b: floating-point maximum value of the distribution
            which_stream: integer value of stream in [0, rng.numStreams() - 1]
        Returns:
            a uniformly generated floating point value in either [a,b) or (a,b)
        '''
        return cls._streams[which_stream].uniform(a,b)


