# dic_spawnre.py

genre_mapping = {
    # =======================================
    # Axx = ROCK
    # =======================================
    'A00': {
        'Hex': '0x00',
        'Genre': 'rock',
        'Parent': None,
        'Related': []
    },
    'A01': {
        'Hex': '0x01',
        'Genre': 'classic rock',
        'Parent': 'A00',
        'Related': []
    },
    'A02': {
        'Hex': '0x02',
        'Genre': 'alternative rock',
        'Parent': 'A00',
        'Related': ['A10']  # grunge
    },
    'A03': {
        'Hex': '0x03',
        'Genre': 'indie rock',
        'Parent': 'A00',
        'Related': ['C03']  # indie pop
    },
    'A04': {
        'Hex': '0x04',
        'Genre': 'folk rock',
        'Parent': 'A00',
        'Related': ['B00', 'B06', 'B07', 'A20']  # folk, acoustic folk, piano folk, country rock
    },
    'A05': {
        'Hex': '0x05',
        'Genre': 'mellow rock',
        'Parent': 'A00',
        'Related': []
    },
    'A06': {
        'Hex': '0x06',
        'Genre': 'acoustic rock',
        'Parent': 'A00',
        'Related': []
    },
    'A07': {
        'Hex': '0x07',
        'Genre': 'piano rock',
        'Parent': 'A00',
        'Related': ['B07']  # piano folk
    },
    'A08': {
        'Hex': '0x08',
        'Genre': 'pop rock',
        'Parent': 'A00',
        'Related': []
    },
    'A09': {
        'Hex': '0x09',
        'Genre': 'hard rock',
        'Parent': 'A00',
        'Related': ['A10', 'A11', 'A12']  # grunge, metal, hardcore
    },
    'A10': {
        'Hex': '0x0A',
        'Genre': 'grunge',
        'Parent': 'A00',
        'Related': ['A02', 'A09', 'A11', 'A12']  # alternative rock, hard rock, metal, hardcore
    },
    'A11': {
        'Hex': '0x0B',
        'Genre': 'metal',
        'Parent': 'A00',
        'Related': []
    },
    'A12': {
        'Hex': '0x0C',
        'Genre': 'hardcore',
        'Parent': 'A00',
        'Related': []
    },
    'A13': {
        'Hex': '0x0D',
        'Genre': 'emo',
        'Parent': 'A00',
        'Related': ['A02']  # alternative rock
    },
    'A14': {
        'Hex': '0x0E',
        'Genre': 'jam band',
        'Parent': 'A00',
        'Related': []
    },
    'A15': {
        'Hex': '0x0F',
        'Genre': 'ska punk',
        'Parent': 'A00',
        'Related': ['A16']  # punk
    },
    'A16': {
        'Hex': '0x10',
        'Genre': 'punk',
        'Parent': 'A00',
        'Related': ['A15', 'C16']  # ska punk, pop punk
    },
    'A17': {
        'Hex': '0x11',
        'Genre': 'surf rock',
        'Parent': 'A00',
        'Related': []
    },
    'A18': {
        'Hex': '0x12',
        'Genre': 'funk rock',
        'Parent': 'A00',
        'Related': []
    },
    'A19': {
        'Hex': '0x13',
        'Genre': 'rock & roll',
        'Parent': 'A00',
        'Related': []
    },
    'A20': {
        'Hex': '0x14',
        'Genre': 'country rock',
        'Parent': ['A00', 'G00'],
        'Related': ['B00']  # folk
    },
    'A21': {
        'Hex': '0x15',
        'Genre': 'blues rock',
        'Parent': ['A00', 'H00'],
        'Related': []
    },
    'A22': {
        'Hex': '0x16',
        'Genre': 'rap rock',
        'Parent': ['A00', 'I00'],
        'Related': []
    },
    'A23': {
        'Hex': '0x17',
        'Genre': 'rock electronica',
        'Parent': ['A00', 'J00'],
        'Related': []
    },

    # =======================================
    # Bxx = FOLK
    # =======================================
    'B00': {
        'Hex': '0x18',
        'Genre': 'folk',
        'Parent': None,
        'Related': ['A04']  # folk rock
    },
    'B01': {
        'Hex': '0x19',
        'Genre': 'singer-songwriter',
        'Parent': 'B00',
        'Related': []
    },
    'B02': {
        'Hex': '0x1A',
        'Genre': 'world music',
        'Parent': 'B00',
        'Related': []
    },
    'B03': {
        'Hex': '0x1B',
        'Genre': 'bluegrass',
        'Parent': ['B00', 'C00'],
        'Related': []
    },
    'B04': {
        'Hex': '0x1C',
        'Genre': 'americana',
        'Parent': 'B00',
        'Related': []
    },
    'B05': {
        'Hex': '0x1D',
        'Genre': 'celtic folk',
        'Parent': 'B00',
        'Related': []
    },
    'B06': {
        'Hex': '0x1E',
        'Genre': 'acoustic folk',
        'Parent': 'B00',
        'Related': []
    },
    'B07': {
        'Hex': '0x1F',
        'Genre': 'piano folk',
        'Parent': 'B00',
        'Related': []
    },
    'B08': {
        'Hex': '0x20',
        'Genre': 'traditional folk',
        'Parent': 'B00',
        'Related': []
    },
    'B09': {
        'Hex': '0x21',
        'Genre': 'folk pop',
        'Parent': ['B00', 'C00'],
        'Related': []
    },
    'B10': {
        'Hex': '0x22',
        'Genre': 'psychedelic folk',
        'Parent': 'B00',
        'Related': []
    },
    'B11': {
        'Hex': '0x23',
        'Genre': 'folk metal',
        'Parent': 'B00',
        'Related': ['A11']  # metal
    },
    'B12': {
        'Hex': '0x24',
        'Genre': 'neofolk',
        'Parent': 'B00',
        'Related': []
    },
    'B13': {
        'Hex': '0x25',
        'Genre': 'folk punk',
        'Parent': 'B00',
        'Related': ['A16']  # punk
    },
    'B14': {
        'Hex': '0x26',
        'Genre': 'folk country',
        'Parent': ['B00', 'G00'],
        'Related': []
    },
    'B15': {
        'Hex': '0x27',
        'Genre': 'folk jazz',
        'Parent': ['B00', 'D00'],
        'Related': []
    },
    'B16': {
        'Hex': '0x28',
        'Genre': 'progressive folk',
        'Parent': 'B00',
        'Related': []
    },
    'B17': {
        'Hex': '0x29',
        'Genre': 'indie folk',
        'Parent': 'B00',
        'Related': ['A03', 'C03']  # indie rock, indie pop
    },
    'B18': {
        'Hex': '0x2A',
        'Genre': 'cajun',
        'Parent': 'B00',
        'Related': ['B02']  # world music
    },
    'B19': {
        'Hex': '0x2B',
        'Genre': 'zydeco',
        'Parent': 'B00',
        'Related': ['B02']  # world music
    },
    'B20': {
        'Hex': '0x2C',
        'Genre': 'anime (music)',
        'Parent': 'B00',
        'Related': []
    },
    'B21': {
        'Hex': '0x2D',
        'Genre': 'bollywood',
        'Parent': 'B00',
        'Related': []
    },
    'B22': {
        'Hex': '0x2E',
        'Genre': 'samba',
        'Parent': 'B00',
        'Related': []
    },
    'B23': {
        'Hex': '0x2F',
        'Genre': '',
        'Parent': 'B00',
        'Related': []
    },

    # =======================================
    # Cxx = POP
    # =======================================
    'C00': {
        'Hex': '0x30',
        'Genre': 'pop',
        'Parent': None,
        'Related': []
    },
    'C01': {
        'Hex': '0x31',
        'Genre': 'dance pop',
        'Parent': 'C00',
        'Related': []
    },
    'C02': {
        'Hex': '0x32',
        'Genre': 'synth-pop',
        'Parent': ['C00', 'J00'],
        'Related': []
    },
    'C03': {
        'Hex': '0x33',
        'Genre': 'indie pop',
        'Parent': 'C00',
        'Related': ['A03']  # indie rock
    },
    'C04': {
        'Hex': '0x34',
        'Genre': 'teen pop',
        'Parent': 'C00',
        'Related': []
    },
    'C05': {
        'Hex': '0x35',
        'Genre': 'electropop',
        'Parent': ['C00', 'J00'],
        'Related': []
    },
    'C06': {
        'Hex': '0x36',
        'Genre': 'dream pop',
        'Parent': 'C00',
        'Related': []
    },
    'C07': {
        'Hex': '0x37',
        'Genre': 'bubblegum pop',
        'Parent': 'C00',
        'Related': []
    },
    'C08': {
        'Hex': '0x38',
        'Genre': 'pop soul',
        'Parent': 'C00',
        'Related': ['F00']  # r&b
    },
    'C09': {
        'Hex': '0x39',
        'Genre': 'power pop',
        'Parent': 'C00',
        'Related': []
    },
    'C10': {
        'Hex': '0x3A',
        'Genre': 'adult contemporary',
        'Parent': 'C00',
        'Related': []
    },
    'C11': {
        'Hex': '0x3B',
        'Genre': 'k-pop',
        'Parent': 'C00',
        'Related': []
    },
    'C12': {
        'Hex': '0x3C',
        'Genre': 'j-pop',
        'Parent': 'C00',
        'Related': []
    },
    'C13': {
        'Hex': '0x3D',
        'Genre': 'c-pop',
        'Parent': 'C00',
        'Related': []
    },
    'C14': {
        'Hex': '0x3E',
        'Genre': 'live',
        'Parent': 'C00',
        'Related': []
    },
    'C15': {
        'Hex': '0x3F',
        'Genre': '',
        'Parent': 'C00',
        'Related': []
    },
    'C16': {
        'Hex': '0x40',
        'Genre': 'pop punk',
        'Parent': ['C00', 'A00'],
        'Related': ['A15', 'A16']  # ska punk, punk
    },
    'C17': {
        'Hex': '0x41',
        'Genre': '',
        'Parent': 'C00',
        'Related': []
    },
    'C18': {
        'Hex': '0x42',
        'Genre': '',
        'Parent': 'C00',
        'Related': []
    },
    'C19': {
        'Hex': '0x43',
        'Genre': '',
        'Parent': 'C00',
        'Related': []
    },
    'C20': {
        'Hex': '0x44',
        'Genre': '',
        'Parent': 'C00',
        'Related': []
    },
    'C21': {
        'Hex': '0x45',
        'Genre': '',
        'Parent': 'C00',
        'Related': []
    },
    'C22': {
        'Hex': '0x46',
        'Genre': '',
        'Parent': 'C00',
        'Related': []
    },
    'C23': {
        'Hex': '0x47',
        'Genre': '',
        'Parent': 'C00',
        'Related': []
    },

    # =======================================
    # Dxx = JAZZ
    # =======================================
    'D00': {
        'Hex': '0x48',
        'Genre': 'jazz',
        'Parent': None,
        'Related': []
    },
    'D01': {
        'Hex': '0x49',
        'Genre': 'vocal jazz',
        'Parent': 'D00',
        'Related': []
    },
    'D02': {
        'Hex': '0x4A',
        'Genre': 'swing',
        'Parent': 'D00',
        'Related': []
    },
    'D03': {
        'Hex': '0x4B',
        'Genre': 'bebop',
        'Parent': 'D00',
        'Related': []
    },
    'D04': {
        'Hex': '0x4C',
        'Genre': 'cool jazz',
        'Parent': 'D00',
        'Related': []
    },
    'D05': {
        'Hex': '0x4D',
        'Genre': 'hard bop',
        'Parent': 'D00',
        'Related': []
    },
    'D06': {
        'Hex': '0x4E',
        'Genre': 'free jazz',
        'Parent': 'D00',
        'Related': []
    },
    'D07': {
        'Hex': '0x4F',
        'Genre': 'fusion',
        'Parent': ['D00', 'A00'],
        'Related': []
    },
    'D08': {
        'Hex': '0x50',
        'Genre': 'jazz pop',
        'Parent': 'D00',
        'Related': []
    },
    'D09': {
        'Hex': '0x51',
        'Genre': 'latin jazz',
        'Parent': 'D00',
        'Related': ['B02']  # world music
    },
    'D10': {
        'Hex': '0x52',
        'Genre': 'smooth jazz',
        'Parent': 'D00',
        'Related': []
    },
    'D11': {
        'Hex': '0x53',
        'Genre': 'acid jazz',
        'Parent': 'D00',
        'Related': []
    },
    'D12': {
        'Hex': '0x54',
        'Genre': 'soul jazz',
        'Parent': 'D00',
        'Related': ['F01']  # soul
    },
    'D13': {
        'Hex': '0x55',
        'Genre': 'bossa nova',
        'Parent': 'D00',
        'Related': []
    },
    'D14': {
        'Hex': '0x56',
        'Genre': 'gypsy jazz',
        'Parent': 'D00',
        'Related': []
    },
    'D15': {
        'Hex': '0x57',
        'Genre': 'dixieland',
        'Parent': 'D00',
        'Related': []
    },
    'D16': {
        'Hex': '0x58',
        'Genre': 'ragtime',
        'Parent': 'D00',
        'Related': []
    },
    'D17': {
        'Hex': '0x59',
        'Genre': 'avant-garde jazz',
        'Parent': 'D00',
        'Related': []
    },
    'D18': {
        'Hex': '0x5A',
        'Genre': 'contemporary jazz',
        'Parent': 'D00',
        'Related': []
    },
    'D19': {
        'Hex': '0x5B',
        'Genre': 'third stream',
        'Parent': 'D00',
        'Related': []
    },
    'D20': {
        'Hex': '0x5C',
        'Genre': 'lounge',
        'Parent': 'D00',
        'Related': []
    },
    'D21': {
        'Hex': '0x5D',
        'Genre': 'big band',
        'Parent': 'D00',
        'Related': []
    },
    'D22': {
        'Hex': '0x5E',
        'Genre': 'jazz rap',
        'Parent': ['D00', 'I00'],
        'Related': []
    },
    # Multi-parent: "electro swing" => Jazz + Electronic
    'D23': {
        'Hex': '0x5F',
        'Genre': 'electro swing',
        'Parent': ['D00', 'J00'],
        'Related': []
    },

    # =======================================
    # Exx = REGGAE / SKA
    # =======================================
    'E00': {
        'Hex': '0x60',
        'Genre': 'reggae',
        'Parent': None,
        'Related': ['E01']  # dub
    },
    'E01': {
        'Hex': '0x61',
        'Genre': 'dub',
        'Parent': 'E00',
        'Related': ['E00']  # reggae
    },
    'E02': {
        'Hex': '0x62',
        'Genre': 'rocksteady',
        'Parent': 'E00',
        'Related': []
    },
    'E03': {
        'Hex': '0x63',
        'Genre': 'dancehall',
        'Parent': 'E00',
        'Related': []
    },
    'E04': {
        'Hex': '0x64',
        'Genre': 'ragga',
        'Parent': 'E00',
        'Related': []
    },
    'E05': {
        'Hex': '0x65',
        'Genre': 'lovers rock',
        'Parent': 'E00',
        'Related': []
    },
    'E06': {
        'Hex': '0x66',
        'Genre': 'roots reggae',
        'Parent': 'E00',
        'Related': []
    },
    'E07': {
        'Hex': '0x67',
        'Genre': '',
        'Parent': 'E00',
        'Related': []
    },
    'E08': {
        'Hex': '0x68',
        'Genre': '',
        'Parent': 'E00',
        'Related': []
    },
    'E09': {
        'Hex': '0x69',
        'Genre': '',
        'Parent': 'E00',
        'Related': []
    },
    'E10': {
        'Hex': '0x6A',
        'Genre': '',
        'Parent': 'E00',
        'Related': []
    },
    'E11': {
        'Hex': '0x6B',
        'Genre': '',
        'Parent': 'E00',
        'Related': []
    },
    'E12': {
        'Hex': '0x6C',
        'Genre': '',
        'Parent': 'E00',
        'Related': []
    },
    'E13': {
        'Hex': '0x6D',
        'Genre': '',
        'Parent': 'E00',
        'Related': []
    },
    'E14': {
        'Hex': '0x6E',
        'Genre': '',
        'Parent': 'E00',
        'Related': []
    },
    'E15': {
        'Hex': '0x6F',
        'Genre': 'ska',
        'Parent': ['E00', 'A00'],
        'Related': ['A16']  # punk
    },
    'E16': {
        'Hex': '0x70',
        'Genre': '',
        'Parent': 'E00',
        'Related': []
    },
    'E17': {
        'Hex': '0x71',
        'Genre': '',
        'Parent': 'E00',
        'Related': []
    },
    'E18': {
        'Hex': '0x72',
        'Genre': '',
        'Parent': 'E00',
        'Related': []
    },
    'E19': {
        'Hex': '0x73',
        'Genre': '',
        'Parent': 'E00',
        'Related': []
    },
    'E20': {
        'Hex': '0x74',
        'Genre': '',
        'Parent': 'E00',
        'Related': []
    },
    'E21': {
        'Hex': '0x75',
        'Genre': '',
        'Parent': 'E00',
        'Related': []
    },

    # =======================================
    # Fxx = R&B / SOUL / FUNK
    # =======================================
    'F00': {
        'Hex': '0x76',
        'Genre': 'r&b',
        'Parent': None,
        'Related': ['F01', 'F18']  # soul, funk
    },
    'F01': {
        'Hex': '0x77',
        'Genre': 'soul',
        'Parent': 'F00',
        'Related': ['F00']
    },
    'F02': {
        'Hex': '0x78',
        'Genre': 'gospel',
        'Parent': 'F00',
        'Related': []
    },
    'F03': {
        'Hex': '0x79',
        'Genre': 'neo-soul',
        'Parent': 'F00',
        'Related': []
    },
    'F04': {
        'Hex': '0x7A',
        'Genre': 'motown',
        'Parent': 'F00',
        'Related': []
    },
    'F05': {
        'Hex': '0x7B',
        'Genre': '',
        'Parent': 'F00',
        'Related': []
    },
    'F06': {
        'Hex': '0x7C',
        'Genre': '',
        'Parent': 'F00',
        'Related': []
    },
    'F07': {
        'Hex': '0x7D',
        'Genre': '',
        'Parent': 'F00',
        'Related': []
    },
    'F08': {
        'Hex': '0x7E',
        'Genre': '',
        'Parent': 'F00',
        'Related': []
    },
    'F09': {
        'Hex': '0x7F',
        'Genre': '',
        'Parent': 'F00',
        'Related': []
    },
    'F10': {
        'Hex': '0x80',
        'Genre': '',
        'Parent': 'F00',
        'Related': []
    },
    'F11': {
        'Hex': '0x81',
        'Genre': '',
        'Parent': 'F00',
        'Related': []
    },
    'F12': {
        'Hex': '0x82',
        'Genre': '',
        'Parent': 'F00',
        'Related': []
    },
    'F13': {
        'Hex': '0x83',
        'Genre': '',
        'Parent': 'F00',
        'Related': []
    },
    'F14': {
        'Hex': '0x84',
        'Genre': '',
        'Parent': 'F00',
        'Related': []
    },
    'F15': {
        'Hex': '0x85',
        'Genre': '',
        'Parent': 'F00',
        'Related': []
    },
    'F16': {
        'Hex': '0x86',
        'Genre': '',
        'Parent': 'F00',
        'Related': []
    },
    'F17': {
        'Hex': '0x87',
        'Genre': '',
        'Parent': 'F00',
        'Related': []
    },
    'F18': {
        'Hex': '0x88',
        'Genre': 'funk',
        'Parent': 'F00',
        'Related': []
    },
    'F19': {
        'Hex': '0x89',
        'Genre': '',
        'Parent': 'F00',
        'Related': []
    },
    'F20': {
        'Hex': '0x8A',
        'Genre': '',
        'Parent': 'F00',
        'Related': []
    },
    'F21': {
        'Hex': '0x8B',
        'Genre': '',
        'Parent': 'F00',
        'Related': []
    },

    # =======================================
    # Gxx = COUNTRY
    # =======================================
    'G00': {
        'Hex': '0x8C',
        'Genre': 'country',
        'Parent': None,
        'Related': ['A20']  # country rock
    },
    'G01': {
        'Hex': '0x8D',
        'Genre': 'country pop',
        'Parent': 'G00',
        'Related': []
    },
    'G02': {
        'Hex': '0x8E',
        'Genre': 'alt-country',
        'Parent': 'G00',
        'Related': []
    },
    'G03': {
        'Hex': '0x8F',
        'Genre': 'honky tonk',
        'Parent': 'G00',
        'Related': []
    },
    'G04': {
        'Hex': '0x90',
        'Genre': '',
        'Parent': 'G00',
        'Related': []
    },
    'G05': {
        'Hex': '0x91',
        'Genre': '',
        'Parent': 'G00',
        'Related': []
    },
    'G06': {
        'Hex': '0x92',
        'Genre': '',
        'Parent': 'G00',
        'Related': []
    },
    'G07': {
        'Hex': '0x93',
        'Genre': '',
        'Parent': 'G00',
        'Related': []
    },
    'G08': {
        'Hex': '0x94',
        'Genre': '',
        'Parent': 'G00',
        'Related': []
    },
    'G09': {
        'Hex': '0x95',
        'Genre': '',
        'Parent': 'G00',
        'Related': []
    },
    'G10': {
        'Hex': '0x96',
        'Genre': '',
        'Parent': 'G00',
        'Related': []
    },
    'G11': {
        'Hex': '0x97',
        'Genre': '',
        'Parent': 'G00',
        'Related': []
    },
    'G12': {
        'Hex': '0x98',
        'Genre': '',
        'Parent': 'G00',
        'Related': []
    },
    'G13': {
        'Hex': '0x99',
        'Genre': '',
        'Parent': 'G00',
        'Related': []
    },
    'G14': {
        'Hex': '0x9A',
        'Genre': '',
        'Parent': 'G00',
        'Related': []
    },
    'G15': {
        'Hex': '0x9B',
        'Genre': '',
        'Parent': 'G00',
        'Related': []
    },
    'G16': {
        'Hex': '0x9C',
        'Genre': '',
        'Parent': 'G00',
        'Related': []
    },
    'G17': {
        'Hex': '0x9D',
        'Genre': '',
        'Parent': 'G00',
        'Related': []
    },
    'G18': {
        'Hex': '0x9E',
        'Genre': '',
        'Parent': 'G00',
        'Related': []
    },
    'G19': {
        'Hex': '0x9F',
        'Genre': '',
        'Parent': 'G00',
        'Related': []
    },
    'G20': {
        'Hex': '0xA0',
        'Genre': '',
        'Parent': 'G00',
        'Related': []
    },
    'G21': {
        'Hex': '0xA1',
        'Genre': '',
        'Parent': 'G00',
        'Related': []
    },

    # =======================================
    # Hxx = BLUES
    # =======================================
    'H00': {
        'Hex': '0xA2',
        'Genre': 'blues',
        'Parent': None,
        'Related': ['A21']  # blues rock
    },
    'H01': {
        'Hex': '0xA3',
        'Genre': 'delta blues',
        'Parent': 'H00',
        'Related': []
    },
    'H02': {
        'Hex': '0xA4',
        'Genre': 'chicago blues',
        'Parent': 'H00',
        'Related': []
    },
    'H03': {
        'Hex': '0xA5',
        'Genre': 'acoustic blues',
        'Parent': 'H00',
        'Related': []
    },
    'H04': {
        'Hex': '0xA6',
        'Genre': 'electric blues',
        'Parent': 'H00',
        'Related': ['H02']  # chicago blues
    },
    'H05': {
        'Hex': '0xA7',
        'Genre': '',
        'Parent': 'H00',
        'Related': []
    },
    'H06': {
        'Hex': '0xA8',
        'Genre': '',
        'Parent': 'H00',
        'Related': []
    },
    'H07': {
        'Hex': '0xA9',
        'Genre': '',
        'Parent': 'H00',
        'Related': []
    },
    'H08': {
        'Hex': '0xAA',
        'Genre': '',
        'Parent': 'H00',
        'Related': []
    },
    'H09': {
        'Hex': '0xAB',
        'Genre': '',
        'Parent': 'H00',
        'Related': []
    },
    'H10': {
        'Hex': '0xAC',
        'Genre': '',
        'Parent': 'H00',
        'Related': []
    },
    'H11': {
        'Hex': '0xAD',
        'Genre': '',
        'Parent': 'H00',
        'Related': []
    },
    'H12': {
        'Hex': '0xAE',
        'Genre': '',
        'Parent': 'H00',
        'Related': []
    },
    'H13': {
        'Hex': '0xAF',
        'Genre': '',
        'Parent': 'H00',
        'Related': []
    },
    'H14': {
        'Hex': '0xB0',
        'Genre': '',
        'Parent': 'H00',
        'Related': []
    },
    'H15': {
        'Hex': '0xB1',
        'Genre': '',
        'Parent': 'H00',
        'Related': []
    },
    'H16': {
        'Hex': '0xB2',
        'Genre': '',
        'Parent': 'H00',
        'Related': []
    },
    'H17': {
        'Hex': '0xB3',
        'Genre': '',
        'Parent': 'H00',
        'Related': []
    },
    'H18': {
        'Hex': '0xB4',
        'Genre': '',
        'Parent': 'H00',
        'Related': []
    },
    'H19': {
        'Hex': '0xB5',
        'Genre': '',
        'Parent': 'H00',
        'Related': []
    },
    'H20': {
        'Hex': '0xB6',
        'Genre': '',
        'Parent': 'H00',
        'Related': []
    },
    'H21': {
        'Hex': '0xB7',
        'Genre': '',
        'Parent': 'H00',
        'Related': []
    },

    # =======================================
    # Ixx = HIP-HOP
    # =======================================
    'I00': {
        'Hex': '0xB8',
        'Genre': 'hip-hop',
        'Parent': None,
        'Related': []
    },
    'I01': {
        'Hex': '0xB9',
        'Genre': 'trap',
        'Parent': 'I00',
        'Related': []
    },
    'I02': {
        'Hex': '0xBA',
        'Genre': 'gangsta rap',
        'Parent': 'I00',
        'Related': []
    },
    'I03': {
        'Hex': '0xBB',
        'Genre': 'old-school hip-hop',
        'Parent': 'I00',
        'Related': []
    },
    'I04': {
        'Hex': '0xBC',
        'Genre': 'alternative hip-hop',
        'Parent': 'I00',
        'Related': []
    },
    'I05': {
        'Hex': '0xBD',
        'Genre': 'east coast rap',
        'Parent': 'I00',
        'Related': []
    },
    'I06': {
        'Hex': '0xBE',
        'Genre': 'west coast rap',
        'Parent': 'I00',
        'Related': []
    },
    'I07': {
        'Hex': '0xBF',
        'Genre': 'dirty south',
        'Parent': 'I00',
        'Related': []
    },
    'I08': {
        'Hex': '0xC0',
        'Genre': 'hardcore rap',
        'Parent': 'I00',
        'Related': []
    },
    'I09': {
        'Hex': '0xC1',
        'Genre': 'latin rap',
        'Parent': 'I00',
        'Related': []
    },
    'I10': {
        'Hex': '0xC2',
        'Genre': 'spoken word',
        'Parent': 'I00',
        'Related': []
    },
    'I11': {
        'Hex': '0xC3',
        'Genre': '',
        'Parent': 'I00',
        'Related': []
    },
    'I12': {
        'Hex': '0xC4',
        'Genre': '',
        'Parent': 'I00',
        'Related': []
    },
    'I13': {
        'Hex': '0xC5',
        'Genre': '',
        'Parent': 'I00',
        'Related': []
    },
    'I14': {
        'Hex': '0xC6',
        'Genre': '',
        'Parent': 'I00',
        'Related': []
    },
    'I15': {
        'Hex': '0xC7',
        'Genre': '',
        'Parent': 'I00',
        'Related': []
    },
    'I16': {
        'Hex': '0xC8',
        'Genre': '',
        'Parent': 'I00',
        'Related': []
    },
    'I17': {
        'Hex': '0xC9',
        'Genre': '',
        'Parent': 'I00',
        'Related': []
    },
    'I18': {
        'Hex': '0xCA',
        'Genre': '',
        'Parent': 'I00',
        'Related': []
    },
    'I19': {
        'Hex': '0xCB',
        'Genre': '',
        'Parent': 'I00',
        'Related': []
    },
    'I20': {
        'Hex': '0xCC',
        'Genre': '',
        'Parent': 'I00',
        'Related': []
    },
    'I21': {
        'Hex': '0xCD',
        'Genre': '',
        'Parent': 'I00',
        'Related': []
    },
    'I22': {
        'Hex': '0xCE',
        'Genre': '',
        'Parent': 'I00',
        'Related': []
    },
    'I23': {
        'Hex': '0xCF',
        'Genre': '',
        'Parent': 'I00',
        'Related': []
    },

    # =======================================
    # Jxx = ELECTRONIC
    # =======================================
    'J00': {
        'Hex': '0xD0',
        'Genre': 'electronic',
        'Parent': None,
        'Related': []
    },
    'J01': {
        'Hex': '0xD1',
        'Genre': 'edm',
        'Parent': 'J00',
        'Related': []
    },
    'J02': {
        'Hex': '0xD2',
        'Genre': 'house',
        'Parent': 'J00',
        'Related': []
    },
    'J03': {
        'Hex': '0xD3',
        'Genre': 'techno',
        'Parent': 'J00',
        'Related': []
    },
    'J04': {
        'Hex': '0xD4',
        'Genre': 'disco',
        'Parent': 'J00',
        'Related': []
    },
    'J05': {
        'Hex': '0xD5',
        'Genre': 'trance',
        'Parent': 'J00',
        'Related': []
    },
    'J06': {
        'Hex': '0xD6',
        'Genre': 'drum and bass',
        'Parent': 'J00',
        'Related': []
    },
    'J07': {
        'Hex': '0xD7',
        'Genre': 'jungle',
        'Parent': 'J00',
        'Related': []
    },
    'J08': {
        'Hex': '0xD8',
        'Genre': 'breakbeat',
        'Parent': 'J00',
        'Related': []
    },
    'J09': {
        'Hex': '0xD9',
        'Genre': 'dubstep',
        'Parent': 'J00',
        'Related': []
    },
    'J10': {
        'Hex': '0xDA',
        'Genre': 'idm',
        'Parent': 'J00',
        'Related': []
    },
    'J11': {
        'Hex': '0xDB',
        'Genre': 'industrial',
        'Parent': 'J00',
        'Related': []
    },
    'J12': {
        'Hex': '0xDC',
        'Genre': '',
        'Parent': 'J00',
        'Related': []
    },
    'J13': {
        'Hex': '0xDD',
        'Genre': '',
        'Parent': 'J00',
        'Related': []
    },
    'J14': {
        'Hex': '0xDE',
        'Genre': '',
        'Parent': 'J00',
        'Related': []
    },
    'J15': {
        'Hex': '0xDF',
        'Genre': '',
        'Parent': 'J00',
        'Related': []
    },
    'J16': {
        'Hex': '0xE0',
        'Genre': '',
        'Parent': 'J00',
        'Related': []
    },
    'J17': {
        'Hex': '0xE1',
        'Genre': '',
        'Parent': 'J00',
        'Related': []
    },
    'J18': {
        'Hex': '0xE2',
        'Genre': '',
        'Parent': 'J00',
        'Related': []
    },
    'J19': {
        'Hex': '0xE3',
        'Genre': '',
        'Parent': 'J00',
        'Related': []
    },
    'J20': {
        'Hex': '0xE4',
        'Genre': '',
        'Parent': 'J00',
        'Related': []
    },
    'J21': {
        'Hex': '0xE5',
        'Genre': '',
        'Parent': 'J00',
        'Related': []
    },
    'J22': {
        'Hex': '0xE6',
        'Genre': '',
        'Parent': 'J00',
        'Related': []
    },
    'J23': {
        'Hex': '0xE7',
        'Genre': '',
        'Parent': 'J00',
        'Related': []
    },

    # =======================================
    # Kxx = CLASSICAL
    # =======================================
    'K00': {
        'Hex': '0xE8',
        'Genre': 'classical',
        'Parent': None,
        'Related': []
    },
    'K01': {
        'Hex': '0xE9',
        'Genre': 'orchestral',
        'Parent': 'K00',
        'Related': []
    },
    'K02': {
        'Hex': '0xEA',
        'Genre': 'opera',
        'Parent': 'K00',
        'Related': []
    },
    'K03': {
        'Hex': '0xEB',
        'Genre': 'baroque',
        'Parent': 'K00',
        'Related': []
    },
    'K04': {
        'Hex': '0xEC',
        'Genre': 'romantic',
        'Parent': 'K00',
        'Related': []
    },
    'K05': {
        'Hex': '0xED',
        'Genre': 'chamber music',
        'Parent': 'K00',
        'Related': []
    },
    'K06': {
        'Hex': '0xEE',
        'Genre': 'choral',
        'Parent': 'K00',
        'Related': []
    },
    'K07': {
        'Hex': '0xEF',
        'Genre': 'piano',
        'Parent': 'K00',
        'Related': []
    },
    'K08': {
        'Hex': '0xF0',
        'Genre': 'modern classical',
        'Parent': 'K00',
        'Related': []
    },
    'K09': {
        'Hex': '0xF1',
        'Genre': 'musical',
        'Parent': 'K00',
        'Related': []
    },
    'K10': {
        'Hex': '0xF2',
        'Genre': '',
        'Parent': 'K00',
        'Related': []
    },
    'K11': {
        'Hex': '0xF3',
        'Genre': '',
        'Parent': 'K00',
        'Related': []
    },
    'K12': {
        'Hex': '0xF4',
        'Genre': '',
        'Parent': 'K00',
        'Related': []
    },
    'K13': {
        'Hex': '0xF5',
        'Genre': '',
        'Parent': 'K00',
        'Related': []
    },
    'K14': {
        'Hex': '0xF6',
        'Genre': '',
        'Parent': 'K00',
        'Related': []
    },
    'K15': {
        'Hex': '0xF7',
        'Genre': '',
        'Parent': 'K00',
        'Related': []
    },
    'K16': {
        'Hex': '0xF8',
        'Genre': '',
        'Parent': 'K00',
        'Related': []
    },
    'K17': {
        'Hex': '0xF9',
        'Genre': '',
        'Parent': 'K00',
        'Related': []
    },
    'K18': {
        'Hex': '0xFA',
        'Genre': '',
        'Parent': 'K00',
        'Related': []
    },
    'K19': {
        'Hex': '0xFB',
        'Genre': '',
        'Parent': 'K00',
        'Related': []
    },
    'K20': {
        'Hex': '0xFC',
        'Genre': '',
        'Parent': 'K00',
        'Related': []
    },
    'K21': {
        'Hex': '0xFD',
        'Genre': '',
        'Parent': 'K00',
        'Related': []
    },
    'K22': {
        'Hex': '0xFE',
        'Genre': '',
        'Parent': 'K00',
        'Related': []
    },
    'K23': {
        'Hex': '0xFF',
        'Genre': '',
        'Parent': 'K00',
        'Related': []
    },
}

subgenre_to_parent = {
    key: details['Parent']
    for key, details in genre_mapping.items()
    if details['Parent'] is not None
}

genre_synonyms = {
    'hiphop': 'hip-hop',
    'hip hop': 'hip-hop',
    'hip-hop/rap': 'hip-hop',
    'rap': 'hip-hop',
    'rhythm & blues': 'r&b',
    'rhythm and blues': 'r&b',
    'rock-n-roll': 'rock & roll',
    'rock and roll': 'rock & roll',
    'punk rock': 'punk',
    'alternative': 'alternative rock'
}

__all__ = ['genre_mapping', 'subgenre_to_parent', 'genre_synonyms']
