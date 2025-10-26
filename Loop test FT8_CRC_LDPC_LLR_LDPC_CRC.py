
import numpy as np
import sys
sys.path.append(r"C:\Users\drala\Documents\Projects\GitHub\PyFT8")
class FT8ref:
    def __init__(self):
        self.url = "https://pengowray.github.io/ft8play/"
        self.msg = "VK1ABC VK3JPK QF22"
        self.bits77 = 0b11100001111111000101001101010111000100000011110100001111000111001010001010001
        self.bits14 = 0b00111100110010
        self.bits83 = 0b01101010111110101110000011111111010100101110011011100110010000000000011100010000001
        self.bits91 = self.bits77 <<14 | self.bits14
        self.bits174 = self.bits91 <<83 | self.bits83

FT8ref = FT8ref()

from decoders.decode174_91_v2_4 import decode174_91

def bitsLE_to_int(bits):
    """bits is MSB-first."""
    n = 0
    for b in bits:
        n = (n << 1) | (b & 1)
    return n

def ldpc_encode(msg_crc: int) -> int:
    generator_matrix_rows = ["8329ce11bf31eaf509f27fc",  "761c264e25c259335493132",  "dc265902fb277c6410a1bdc",  "1b3f417858cd2dd33ec7f62",  "09fda4fee04195fd034783a",  "077cccc11b8873ed5c3d48a",  "29b62afe3ca036f4fe1a9da",  "6054faf5f35d96d3b0c8c3e",  "e20798e4310eed27884ae90",  "775c9c08e80e26ddae56318",  "b0b811028c2bf997213487c",  "18a0c9231fc60adf5c5ea32",  "76471e8302a0721e01b12b8",  "ffbccb80ca8341fafb47b2e",  "66a72a158f9325a2bf67170",  "c4243689fe85b1c51363a18",  "0dff739414d1a1b34b1c270",  "15b48830636c8b99894972e",  "29a89c0d3de81d665489b0e",  "4f126f37fa51cbe61bd6b94",  "99c47239d0d97d3c84e0940",  "1919b75119765621bb4f1e8",  "09db12d731faee0b86df6b8",  "488fc33df43fbdeea4eafb4",  "827423ee40b675f756eb5fe",  "abe197c484cb74757144a9a",  "2b500e4bc0ec5a6d2bdbdd0",  "c474aa53d70218761669360",  "8eba1a13db3390bd6718cec",  "753844673a27782cc42012e",  "06ff83a145c37035a5c1268",  "3b37417858cc2dd33ec3f62",  "9a4a5a28ee17ca9c324842c",  "bc29f465309c977e89610a4",  "2663ae6ddf8b5ce2bb29488",  "46f231efe457034c1814418",  "3fb2ce85abe9b0c72e06fbe",  "de87481f282c153971a0a2e",  "fcd7ccf23c69fa99bba1412",  "f0261447e9490ca8e474cec",  "4410115818196f95cdd7012",  "088fc31df4bfbde2a4eafb4",  "b8fef1b6307729fb0a078c0",  "5afea7acccb77bbc9d99a90",  "49a7016ac653f65ecdc9076",  "1944d085be4e7da8d6cc7d0",  "251f62adc4032f0ee714002",  "56471f8702a0721e00b12b8",  "2b8e4923f2dd51e2d537fa0",  "6b550a40a66f4755de95c26",  "a18ad28d4e27fe92a4f6c84",  "10c2e586388cb82a3d80758",  "ef34a41817ee02133db2eb0",  "7e9c0c54325a9c15836e000",  "3693e572d1fde4cdf079e86",  "bfb2cec5abe1b0c72e07fbe",  "7ee18230c583cccc57d4b08",  "a066cb2fedafc9f52664126",  "bb23725abc47cc5f4cc4cd2",  "ded9dba3bee40c59b5609b4",  "d9a7016ac653e6decdc9036",  "9ad46aed5f707f280ab5fc4",  "e5921c77822587316d7d3c2",  "4f14da8242a8b86dca73352",  "8b8b507ad467d4441df770e",  "22831c9cf1169467ad04b68",  "213b838fe2ae54c38ee7180",  "5d926b6dd71f085181a4e12",  "66ab79d4b29ee6e69509e56",  "958148682d748a38dd68baa",  "b8ce020cf069c32a723ab14",  "f4331d6d461607e95752746",  "6da23ba424b9596133cf9c8",  "a636bcbc7b30c5fbeae67fe",  "5cb0d86a07df654a9089a20",  "f11f106848780fc9ecdd80a",  "1fbb5364fb8d2c9d730d5ba",  "fcb86bc70a50c9d02a5d034",  "a534433029eac15f322e34c",  "c989d9c7c3d3b8c55d75130",  "7bb38b2f0186d46643ae962",  "2644ebadeb44b9467d1f42c",  "608cc857594bfbb55d69600"]
    kGEN = np.array([int(row,16)>>1 for row in generator_matrix_rows])
    msg_crc = int(msg_crc)
    parity_bits = 0
    for row in map(int, kGEN):
        bit = bin(msg_crc & row).count("1") & 1
        parity_bits = (parity_bits << 1) | bit
    return (msg_crc << 83) | parity_bits

def add_noise_to_bits(bits, snr_dB):
    noise = np.random.normal(0.5,0.5,len(bits))
    e_bits = np.sum(bits **2)
    e_noise = np.sum(noise **2)
    snr_init_dB = 10*np.log10(e_bits / e_noise)
    snr_adj = snr_dB - snr_init_dB
    noise = noise * 10**(-snr_adj/20)
    return bits + noise

def crc14(bits77_int: int) -> int:
    # Generator polynomial (0x2757), width 14, init=0, refin=false, refout=false
    poly = 0x2757
    width = 14
    mask = (1 << width) - 1
    # Pad to 96 bits (77 + 14 + 5)
    nbits = 96
    reg_int = 0
    for i in range(nbits):
        # bits77 is expected MSB-first (bit 76 first)
        inbit = ((bits77_int >> (76 - i)) & 1) if i < 77 else 0
        bit14 = (reg_int >> (width - 1)) & 1
        reg_int = ((reg_int << 1) & mask) | inbit
        if bit14:
            reg_int ^= poly
    return reg_int

def append_crc(bits77_int):
    """Append 14-bit WSJT-X CRC to a 77-bit message, returning a 91-bit list."""
    bits14_int = crc14(bits77_int)
    bits91_int = (bits77_int << 14) | bits14_int
    return bits91_int

def check_crc(bits91_int):
    """Return True if the 91-bit message (77 data + 14 CRC) passes WSJT-X CRC-14."""
    bits14_int = bits91_int & 0b11111111111111
    bits77_int = bits91_int >> 14
    return bits14_int == crc14(bits77_int)

def int_to_bitsLE(n, width):
    """Return [b(width-1), ..., b0], MSB-first."""
    return [ (n >> (width - 1 - i)) & 1 for i in range(width) ]

def run_loop(snr_dB):
    bits77_int = FT8ref.bits77
    bits91_int = append_crc(bits77_int)
    bits174_int = ldpc_encode(bits91_int)
    bits174_LE_list = int_to_bitsLE(bits174_int,174)
    bits_plus_noise = add_noise_to_bits(np.array(bits174_LE_list), snr_dB)
    llr = 200000 * bits_plus_noise - 100000
    ncheck, decoded_bits174_LE_list, it = decode174_91(llr)
    decoded_bits174_int = bitsLE_to_int(decoded_bits174_LE_list)
    return bits77_int, bits91_int, bits174_int, llr, decoded_bits174_int, it

def run_loop_print(snr_dB):
    bits77_int, bits91_int, bits174_int, llr, decoded_bits174_int, it = run_loop(snr_dB)
    print(f"77 message bits for 'VK1ABC VK3JPK QF22':\n{bits77_int:b}")
    print(f"Message plus crc (91 bits):\n{bits91_int:b}")
    print(f"CRC loop test: check_crc(append_crc(bits77_int)) = { check_crc(append_crc(bits77_int))}")
    print(f"Message plus crc reference from {FT8ref.url}\n{FT8ref.bits91:b}")
    print(f"With parity (174 bits):\n{bits174_int:b}")
    print(f"Message plus crc and parity reference from {FT8ref.url}\n{FT8ref.bits174:b}")
    #print(f"LLRs with noise:\n"+','.join([f"{ll:.1f} " for ll in llr]))
    if(decoded_bits174_int>0):
        print(f"SNR = {snr_dB}dB. Bits decoded after {it} iterations\n{bits174_int:b}")
    else:
        print(f"SNR = {snr_dB}dB. Decoder stopped after {it} iterations")

def run_trials(n, snr_dB):
    s = 0
    for i in range(n):
        bits77_int, bits91_int, bits174_int, llr, decoded_bits174_int, it = run_loop(snr_dB)
        if(decoded_bits174_int>0):
            s +=1
    return(s/n)

def test_vs_snr():
    import time
    t0=time.time()
    nTrials = 50
    print("snr_dB, success%")
    for snr_dB in np.linspace(5, 8, 10):
        success = run_trials(nTrials, snr_dB)
        print(f"{snr_dB:.1f}, {success:.0%} time = {time.time()-t0:.1f}")

test_vs_snr()

#run_loop_print(8)




