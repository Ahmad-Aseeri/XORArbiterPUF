import random

import numpy as np

import pypuf.io
import pypuf.simulation.delay


class PUFGenerator(object):
    def __init__(self, num_stages, num_streams):
        self.num_stages = num_stages
        self.num_streams = num_streams

    def generate(self, num_challenges, size=1, rank=0):
        num_ch = int(num_challenges / size)
        C = self.generate_challenges(num_ch, rank)
        C, r = self.simulate_challenges(C)
        return C, r

    def generate_challenges(self, num_challenges, rank):
        np.random.seed(random.randint(0, 2 ** 16) + rank)
        C = np.random.choice(np.array([0, 1], dtype=np.int8), size=(num_challenges, self.num_stages))
        challenges = np.unique(C, axis=0) if self.num_stages < 32 else C

        # challenges = [random.randint(0, 2 ** self.num_stages - 1) for _ in range(num_challenges)]
        #
        # # Removes duplicates to avoid using them in training and test simulatenously
        # challenges = list(set(challenges))
        #
        # # Transforms the challenges into their string representation
        # # (this step could be avoided and implemented in a more efficient way)
        # challenges = [('{0:0' + str(self.num_stages) + 'b}').format(x) for x in challenges]
        #
        # # Transforms each bit into an integer.
        # challenges = np.asarray([list(map(int, list(x))) for x in challenges], dtype=np.int8)

        return challenges


    def simulate_challenges(self, challenges):
        responses_list = []
        for challenge in challenges:
            responses_list.append(self.simulate_one_challenge(challenge))

        return challenges, np.asarray(responses_list, dtype=np.int8)


    def simulate_one_challenge(self, challenge):
        raise NotImplementedError

    def random_initialization(self):
        return self



"""
This class implements the Arbiter PUF simulator.
Delays are drawn from a Gaussian distributed with mean 300 and standard
deviation 40.
"""


class LinearPUFGenerator(PUFGenerator):
    def __init__(self, num_stages=64, id=0):
        self.puf_id = id
        self.num_stages = num_stages

        # Generate the random Arbiter PUF
        self.puf_arbiter = np.random.normal(300, 40, size=(self.num_stages, 4))

        # Removes negative delays
        self.puf_arbiter[self.puf_arbiter < 0] = 300.

    def simulate_one_challenge(self, challenge):

        # Current delay of the top and bottom paths.
        delay_top = 0.
        delay_bottom = 0.

        # For each stage in the input challenge.
        for input_value, time_diff in zip(challenge, self.puf_arbiter):
            delay_top0 = delay_top
            if input_value == 1:               # Straight transition
                delay_top += time_diff[0]
                delay_bottom += time_diff[3]
            elif input_value == 0:             # Crossed transition
                delay_top = delay_bottom + time_diff[1]
                delay_bottom = delay_top0 + time_diff[2]

        # # Current side of the top path.
        # top_is_up = True
        # for input_value, time_diff in zip(challenge, self.puf_arbiter):
        #     if top_is_up:
        #         # The top path is in the top side of the arbiter.
        #         if input_value == 1:
        #             # Straight transition
        #             delay_top += time_diff[0]
        #             delay_bottom += time_diff[3]
        #         elif input_value == 0:
        #             # Crossed transition
        #             delay_top += time_diff[1]
        #             delay_bottom += time_diff[2]
        #             top_is_up = not top_is_up
        #     elif not top_is_up:
        #         # The top path is in the bottom side of the arbiter.
        #         if input_value == 1:
        #             # Straight transition
        #             delay_top += time_diff[3]
        #             delay_bottom += time_diff[0]
        #         elif input_value == 0:
        #             # Crossed transition
        #             delay_top += time_diff[2]
        #             delay_bottom += time_diff[1]
        #             top_is_up = not top_is_up

        delay = delay_top - delay_bottom    # Negative value implies faster signal to reach top
        if delay < 0.0:
            return 1
        else:
            return 0

    def simulate_challenges(self, challenges):
        # shorthands
        n = self.num_stages
        dTT = self.puf_arbiter[:, 0]
        dBB = self.puf_arbiter[:, 3]
        dBT = self.puf_arbiter[:, 1]
        dTB = self.puf_arbiter[:, 2]

        # Compute weights and bias for evaluation using a linear threshold function.
        # The formula used below can be obtained using an inductive proof over the
        # number of stages in each Arbiter PUF.
        weights = np.empty(shape=(n,))
        weights[0] = .5 * (dTT[0] - dBB[0] - dBT[0] + dTB[0])
        weights[0] *= 1
        for i in range(1, n):
            weights[i] = -.5 * (-1)**(i-n) * (
                - dTT[i-1] + dTB[i-1] - dBT[i-1] + dBB[i-1]
                - dTT[i] - dTB[i] + dBT[i] + dBB[i]
            )
        bias = .5 * (dTT[n-1] - dBB[n-1] + dBT[n-1] - dTB[n-1])

        # use pypuf's LTF simulator to compute responses fast
        simulator = pypuf.simulation.delay.LTFArray(
            weight_array=weights.reshape((1, n)),
            bias=bias,
            transform=pypuf.simulation.delay.XORArbiterPUF.transform_atf,
            combiner=pypuf.simulation.delay.LTFArray.combiner_xor,
        )

        # pypuf only understands challenges and responses in {-1,1} format,
        # hence the conversion
        return challenges, (simulator.eval(1 - 2 * challenges) == -1).astype(np.int8)


"""
This class implements the XOR Arbiter PUF simulator. Each stream is generated
in an independent way using the LinearPUFGenerator.
"""


class XORPUFGenerator(PUFGenerator):
    def __init__(self, num_stages=64, num_streams=2, num_challenges=10):
        super(XORPUFGenerator, self).__init__(num_stages, num_streams)
        self.num_stages = num_stages
        self.num_streams = num_streams
        self.num_challenges = num_challenges
        self.iterations = 1
        self.arbiter_initialization()

    def arbiter_initialization(self):
        # An XOR Arbiter PUF is formed by N independent streams and a final
        # XOR function. This variable stores the delays for the N streams.
        self.linear_pufs = []
        for i in range(self.num_streams):
            self.linear_pufs.append(LinearPUFGenerator(self.num_stages, i))
        return self

    def simulate_one_challenge(self, challenge):

        results = []
        for lpuf in self.linear_pufs:
            results.append(lpuf.simulate_one_challenge(challenge))

        # Apply the XOR function
        results = [r == 1 for r in results]
        r = results[0]
        for next_result in results[1:]:
            r = r != next_result

        if r:
            return 1
        else:
            return 0

    def simulate_challenges(self, challenges):
        return challenges, (np.sum([p.simulate_challenges(challenges)[1] for p in self.linear_pufs], axis=0) % 2).astype(np.int8)


def test_arbiter_puf_simulation():
    """Tests if delay-based and pypuf-based simulation gives the same result in `LinearPUFGenerator`."""
    puf = LinearPUFGenerator(64)
    challenges = puf.generate_challenges(num_challenges=100, rank=0)
    _, r1 = PUFGenerator.simulate_challenges(puf, challenges)
    _, r2 = puf.simulate_challenges(challenges)
    assert np.mean(r1 == r2) == 1


def test_xor_arbiter_puf_simulation():
    """Tests if delay-based and pypuf-based simulation gives the same result in `XORPUFGenerator`."""
    puf = XORPUFGenerator(64, 3)
    challenges = puf.generate_challenges(num_challenges=100, rank=0)
    _, r1 = PUFGenerator.simulate_challenges(puf, challenges)
    _, r2 = puf.simulate_challenges(challenges)
    assert np.mean(r1 == r2) == 1
