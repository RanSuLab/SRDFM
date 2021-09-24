import gc

import numpy as np

import pandas as pd

import config2 as config

import random


class Id_Index(object):

    def __index__(self,id,index):
        self.id = id
        self.index = index

    def getId(self):
        return self.id

    def getIndex(self):
        return self.index



class FeatureDictionary(object):

    def __init__(self, catefile=None,numeric_cols = None,ignore_cols = None,

                 cate_cols=[]):


        # self.trainfile = trainfile

        self.catefile = catefile

        self.cate_cols = cate_cols

        self.numeric_cols = numeric_cols

        self.ignore_cols = ignore_cols


        if self.cate_cols:

            self.gen_feat_dict()

        else:

            self.feat_dict = {'PubchemFP0': {0: 0, 1: 1}, 'PubchemFP1': {0: 2, 1: 3}, 'PubchemFP2': {0: 4, 1: 5},
                              'PubchemFP3': {0: 6, 1: 7}, 'PubchemFP4': {0: 8}, 'PubchemFP5': {0: 9},
                              'PubchemFP6': {0: 10, 1: 11}, 'PubchemFP7': {0: 12}, 'PubchemFP8': {0: 13},
                              'PubchemFP9': {1: 14}, 'PubchemFP10': {1: 15}, 'PubchemFP11': {0: 16, 1: 17},
                              'PubchemFP12': {0: 18, 1: 19}, 'PubchemFP13': {0: 20, 1: 21},
                              'PubchemFP14': {0: 22, 1: 23}, 'PubchemFP15': {0: 24, 1: 25},
                              'PubchemFP16': {0: 26, 1: 27}, 'PubchemFP17': {0: 28, 1: 29},
                              'PubchemFP18': {0: 30, 1: 31}, 'PubchemFP19': {0: 32, 1: 33},
                              'PubchemFP20': {0: 34, 1: 35}, 'PubchemFP21': {0: 36, 1: 37},
                              'PubchemFP22': {0: 38, 1: 39}, 'PubchemFP23': {0: 40, 1: 41},
                              'PubchemFP24': {0: 42, 1: 43}, 'PubchemFP25': {0: 44, 1: 45}, 'PubchemFP26': {0: 46},
                              'PubchemFP27': {0: 47}, 'PubchemFP28': {0: 48}, 'PubchemFP29': {0: 49},
                              'PubchemFP30': {0: 50, 1: 51}, 'PubchemFP31': {0: 52}, 'PubchemFP32': {0: 53},
                              'PubchemFP33': {0: 54, 1: 55}, 'PubchemFP34': {0: 56, 1: 57}, 'PubchemFP35': {0: 58},
                              'PubchemFP36': {0: 59}, 'PubchemFP37': {0: 60, 1: 61}, 'PubchemFP38': {0: 62, 1: 63},
                              'PubchemFP39': {0: 64}, 'PubchemFP40': {0: 65}, 'PubchemFP41': {0: 66},
                              'PubchemFP42': {0: 67}, 'PubchemFP43': {0: 68, 1: 69}, 'PubchemFP44': {0: 70, 1: 71},
                              'PubchemFP45': {0: 72}, 'PubchemFP46': {0: 73, 1: 74}, 'PubchemFP47': {0: 75},
                              'PubchemFP48': {0: 76}, 'PubchemFP49': {0: 77}, 'PubchemFP50': {0: 78},
                              'PubchemFP51': {0: 79}, 'PubchemFP52': {0: 80}, 'PubchemFP53': {0: 81},
                              'PubchemFP54': {0: 82}, 'PubchemFP55': {0: 83}, 'PubchemFP56': {0: 84},
                              'PubchemFP57': {0: 85}, 'PubchemFP58': {0: 86}, 'PubchemFP59': {0: 87},
                              'PubchemFP60': {0: 88}, 'PubchemFP61': {0: 89}, 'PubchemFP62': {0: 90},
                              'PubchemFP63': {0: 91}, 'PubchemFP64': {0: 92}, 'PubchemFP65': {0: 93},
                              'PubchemFP66': {0: 94}, 'PubchemFP67': {0: 95}, 'PubchemFP68': {0: 96},
                              'PubchemFP69': {0: 97}, 'PubchemFP70': {0: 98}, 'PubchemFP71': {0: 99},
                              'PubchemFP72': {0: 100}, 'PubchemFP73': {0: 101}, 'PubchemFP74': {0: 102},
                              'PubchemFP75': {0: 103}, 'PubchemFP76': {0: 104}, 'PubchemFP77': {0: 105},
                              'PubchemFP78': {0: 106}, 'PubchemFP79': {0: 107}, 'PubchemFP80': {0: 108},
                              'PubchemFP81': {0: 109}, 'PubchemFP82': {0: 110}, 'PubchemFP83': {0: 111},
                              'PubchemFP84': {0: 112}, 'PubchemFP85': {0: 113}, 'PubchemFP86': {0: 114},
                              'PubchemFP87': {0: 115}, 'PubchemFP88': {0: 116}, 'PubchemFP89': {0: 117},
                              'PubchemFP90': {0: 118}, 'PubchemFP91': {0: 119}, 'PubchemFP92': {0: 120},
                              'PubchemFP93': {0: 121}, 'PubchemFP94': {0: 122}, 'PubchemFP95': {0: 123},
                              'PubchemFP96': {0: 124}, 'PubchemFP97': {0: 125}, 'PubchemFP98': {0: 126},
                              'PubchemFP99': {0: 127}, 'PubchemFP100': {0: 128}, 'PubchemFP101': {0: 129},
                              'PubchemFP102': {0: 130}, 'PubchemFP103': {0: 131}, 'PubchemFP104': {0: 132},
                              'PubchemFP105': {0: 133}, 'PubchemFP106': {0: 134}, 'PubchemFP107': {0: 135},
                              'PubchemFP108': {0: 136}, 'PubchemFP109': {0: 137}, 'PubchemFP110': {0: 138},
                              'PubchemFP111': {0: 139}, 'PubchemFP112': {0: 140}, 'PubchemFP113': {0: 141},
                              'PubchemFP114': {0: 142}, 'PubchemFP115': {0: 143, 1: 144},
                              'PubchemFP116': {0: 145, 1: 146}, 'PubchemFP117': {0: 147, 1: 148},
                              'PubchemFP118': {0: 149, 1: 150}, 'PubchemFP119': {0: 151}, 'PubchemFP120': {0: 152},
                              'PubchemFP121': {0: 153}, 'PubchemFP122': {0: 154}, 'PubchemFP123': {0: 155},
                              'PubchemFP124': {0: 156}, 'PubchemFP125': {0: 157}, 'PubchemFP126': {0: 158},
                              'PubchemFP127': {0: 159}, 'PubchemFP128': {0: 160}, 'PubchemFP129': {0: 161, 1: 162},
                              'PubchemFP130': {0: 163, 1: 164}, 'PubchemFP131': {0: 165},
                              'PubchemFP132': {0: 166, 1: 167}, 'PubchemFP133': {0: 168}, 'PubchemFP134': {0: 169},
                              'PubchemFP135': {0: 170}, 'PubchemFP136': {0: 171}, 'PubchemFP137': {0: 172},
                              'PubchemFP138': {0: 173}, 'PubchemFP139': {0: 174}, 'PubchemFP140': {0: 175},
                              'PubchemFP141': {0: 176}, 'PubchemFP142': {0: 177}, 'PubchemFP143': {0: 178, 1: 179},
                              'PubchemFP144': {0: 180, 1: 181}, 'PubchemFP145': {0: 182, 1: 183},
                              'PubchemFP146': {0: 184, 1: 185}, 'PubchemFP147': {0: 186, 1: 187},
                              'PubchemFP148': {0: 188, 1: 189}, 'PubchemFP149': {0: 190, 1: 191},
                              'PubchemFP150': {0: 192, 1: 193}, 'PubchemFP151': {0: 194, 1: 195},
                              'PubchemFP152': {0: 196, 1: 197}, 'PubchemFP153': {0: 198, 1: 199},
                              'PubchemFP154': {0: 200}, 'PubchemFP155': {0: 201, 1: 202},
                              'PubchemFP156': {0: 203, 1: 204}, 'PubchemFP157': {0: 205, 1: 206},
                              'PubchemFP158': {0: 207}, 'PubchemFP159': {0: 208, 1: 209},
                              'PubchemFP160': {0: 210, 1: 211}, 'PubchemFP161': {0: 212}, 'PubchemFP162': {0: 213},
                              'PubchemFP163': {0: 214}, 'PubchemFP164': {0: 215, 1: 216}, 'PubchemFP165': {0: 217},
                              'PubchemFP166': {0: 218, 1: 219}, 'PubchemFP167': {0: 220, 1: 221},
                              'PubchemFP168': {0: 222}, 'PubchemFP169': {0: 223}, 'PubchemFP170': {0: 224},
                              'PubchemFP171': {0: 225}, 'PubchemFP172': {0: 226}, 'PubchemFP173': {0: 227},
                              'PubchemFP174': {0: 228}, 'PubchemFP175': {0: 229}, 'PubchemFP176': {0: 230},
                              'PubchemFP177': {0: 231}, 'PubchemFP178': {0: 232, 1: 233},
                              'PubchemFP179': {0: 234, 1: 235}, 'PubchemFP180': {0: 236, 1: 237},
                              'PubchemFP181': {0: 238, 1: 239}, 'PubchemFP182': {0: 240, 1: 241},
                              'PubchemFP183': {0: 242, 1: 243}, 'PubchemFP184': {0: 244, 1: 245},
                              'PubchemFP185': {0: 246, 1: 247}, 'PubchemFP186': {0: 248, 1: 249},
                              'PubchemFP187': {0: 250, 1: 251}, 'PubchemFP188': {0: 252, 1: 253},
                              'PubchemFP189': {0: 254, 1: 255}, 'PubchemFP190': {0: 256, 1: 257},
                              'PubchemFP191': {0: 258, 1: 259}, 'PubchemFP192': {0: 260, 1: 261},
                              'PubchemFP193': {0: 262, 1: 263}, 'PubchemFP194': {0: 264, 1: 265},
                              'PubchemFP195': {0: 266, 1: 267}, 'PubchemFP196': {0: 268}, 'PubchemFP197': {0: 269},
                              'PubchemFP198': {0: 270}, 'PubchemFP199': {0: 271, 1: 272},
                              'PubchemFP200': {0: 273, 1: 274}, 'PubchemFP201': {0: 275, 1: 276},
                              'PubchemFP202': {0: 277, 1: 278}, 'PubchemFP203': {0: 279}, 'PubchemFP204': {0: 280},
                              'PubchemFP205': {0: 281}, 'PubchemFP206': {0: 282, 1: 283}, 'PubchemFP207': {0: 284},
                              'PubchemFP208': {0: 285}, 'PubchemFP209': {0: 286}, 'PubchemFP210': {0: 287},
                              'PubchemFP211': {0: 288}, 'PubchemFP212': {0: 289}, 'PubchemFP213': {0: 290, 1: 291},
                              'PubchemFP214': {0: 292, 1: 293}, 'PubchemFP215': {0: 294, 1: 295},
                              'PubchemFP216': {0: 296, 1: 297}, 'PubchemFP217': {0: 298},
                              'PubchemFP218': {0: 299, 1: 300}, 'PubchemFP219': {0: 301, 1: 302},
                              'PubchemFP220': {0: 303}, 'PubchemFP221': {0: 304}, 'PubchemFP222': {0: 305},
                              'PubchemFP223': {0: 306}, 'PubchemFP224': {0: 307}, 'PubchemFP225': {0: 308},
                              'PubchemFP226': {0: 309}, 'PubchemFP227': {0: 310, 1: 311},
                              'PubchemFP228': {0: 312, 1: 313}, 'PubchemFP229': {0: 314, 1: 315},
                              'PubchemFP230': {0: 316, 1: 317}, 'PubchemFP231': {0: 318}, 'PubchemFP232': {0: 319},
                              'PubchemFP233': {0: 320}, 'PubchemFP234': {0: 321}, 'PubchemFP235': {0: 322},
                              'PubchemFP236': {0: 323}, 'PubchemFP237': {0: 324}, 'PubchemFP238': {0: 325},
                              'PubchemFP239': {0: 326}, 'PubchemFP240': {0: 327}, 'PubchemFP241': {0: 328, 1: 329},
                              'PubchemFP242': {0: 330}, 'PubchemFP243': {0: 331, 1: 332},
                              'PubchemFP244': {0: 333, 1: 334}, 'PubchemFP245': {0: 335}, 'PubchemFP246': {0: 336},
                              'PubchemFP247': {0: 337}, 'PubchemFP248': {0: 338, 1: 339}, 'PubchemFP249': {0: 340},
                              'PubchemFP250': {0: 341}, 'PubchemFP251': {0: 342}, 'PubchemFP252': {0: 343, 1: 344},
                              'PubchemFP253': {0: 345}, 'PubchemFP254': {0: 346}, 'PubchemFP255': {0: 347, 1: 348},
                              'PubchemFP256': {0: 349, 1: 350}, 'PubchemFP257': {0: 351, 1: 352},
                              'PubchemFP258': {0: 353, 1: 354}, 'PubchemFP259': {0: 355, 1: 356},
                              'PubchemFP260': {0: 357, 1: 358}, 'PubchemFP261': {0: 359, 1: 360},
                              'PubchemFP262': {0: 361, 1: 362}, 'PubchemFP263': {0: 363}, 'PubchemFP264': {0: 364},
                              'PubchemFP265': {0: 365}, 'PubchemFP266': {0: 366}, 'PubchemFP267': {0: 367},
                              'PubchemFP268': {0: 368}, 'PubchemFP269': {0: 369}, 'PubchemFP270': {0: 370},
                              'PubchemFP271': {0: 371}, 'PubchemFP272': {0: 372}, 'PubchemFP273': {0: 373},
                              'PubchemFP274': {0: 374, 1: 375}, 'PubchemFP275': {0: 376},
                              'PubchemFP276': {0: 377, 1: 378}, 'PubchemFP277': {0: 379}, 'PubchemFP278': {0: 380},
                              'PubchemFP279': {0: 381}, 'PubchemFP280': {0: 382}, 'PubchemFP281': {0: 383},
                              'PubchemFP282': {0: 384}, 'PubchemFP283': {1: 385}, 'PubchemFP284': {1: 386},
                              'PubchemFP285': {0: 387, 1: 388}, 'PubchemFP286': {0: 389, 1: 390},
                              'PubchemFP287': {0: 391, 1: 392}, 'PubchemFP288': {0: 393}, 'PubchemFP289': {0: 394},
                              'PubchemFP290': {0: 395}, 'PubchemFP291': {0: 396}, 'PubchemFP292': {0: 397},
                              'PubchemFP293': {0: 398, 1: 399}, 'PubchemFP294': {0: 400, 1: 401},
                              'PubchemFP295': {0: 402}, 'PubchemFP296': {0: 403}, 'PubchemFP297': {0: 404, 1: 405},
                              'PubchemFP298': {0: 406, 1: 407}, 'PubchemFP299': {0: 408, 1: 409},
                              'PubchemFP300': {0: 410, 1: 411}, 'PubchemFP301': {0: 412, 1: 413},
                              'PubchemFP302': {0: 414}, 'PubchemFP303': {0: 415}, 'PubchemFP304': {0: 416},
                              'PubchemFP305': {0: 417, 1: 418}, 'PubchemFP306': {0: 419}, 'PubchemFP307': {0: 420},
                              'PubchemFP308': {0: 421, 1: 422}, 'PubchemFP309': {0: 423}, 'PubchemFP310': {0: 424},
                              'PubchemFP311': {0: 425}, 'PubchemFP312': {0: 426}, 'PubchemFP313': {0: 427},
                              'PubchemFP314': {0: 428, 1: 429}, 'PubchemFP315': {0: 430}, 'PubchemFP316': {0: 431},
                              'PubchemFP317': {0: 432}, 'PubchemFP318': {0: 433}, 'PubchemFP319': {0: 434},
                              'PubchemFP320': {0: 435}, 'PubchemFP321': {0: 436}, 'PubchemFP322': {0: 437},
                              'PubchemFP323': {0: 438}, 'PubchemFP324': {0: 439}, 'PubchemFP325': {0: 440},
                              'PubchemFP326': {0: 441}, 'PubchemFP327': {0: 442, 1: 443},
                              'PubchemFP328': {0: 444, 1: 445}, 'PubchemFP329': {0: 446},
                              'PubchemFP330': {0: 447, 1: 448}, 'PubchemFP331': {0: 449},
                              'PubchemFP332': {0: 450, 1: 451}, 'PubchemFP333': {0: 452, 1: 453},
                              'PubchemFP334': {0: 454, 1: 455}, 'PubchemFP335': {0: 456, 1: 457},
                              'PubchemFP336': {0: 458, 1: 459}, 'PubchemFP337': {0: 460, 1: 461},
                              'PubchemFP338': {0: 462, 1: 463}, 'PubchemFP339': {0: 464, 1: 465},
                              'PubchemFP340': {0: 466, 1: 467}, 'PubchemFP341': {0: 468, 1: 469},
                              'PubchemFP342': {0: 470, 1: 471}, 'PubchemFP343': {0: 472, 1: 473},
                              'PubchemFP344': {0: 474, 1: 475}, 'PubchemFP345': {0: 476, 1: 477},
                              'PubchemFP346': {0: 478, 1: 479}, 'PubchemFP347': {0: 480, 1: 481},
                              'PubchemFP348': {0: 482}, 'PubchemFP349': {0: 483, 1: 484},
                              'PubchemFP350': {0: 485, 1: 486}, 'PubchemFP351': {0: 487, 1: 488},
                              'PubchemFP352': {0: 489, 1: 490}, 'PubchemFP353': {0: 491, 1: 492},
                              'PubchemFP354': {0: 493}, 'PubchemFP355': {0: 494, 1: 495},
                              'PubchemFP356': {0: 496, 1: 497}, 'PubchemFP357': {0: 498, 1: 499},
                              'PubchemFP358': {0: 500, 1: 501}, 'PubchemFP359': {0: 502, 1: 503},
                              'PubchemFP360': {0: 504, 1: 505}, 'PubchemFP361': {0: 506, 1: 507},
                              'PubchemFP362': {0: 508, 1: 509}, 'PubchemFP363': {0: 510, 1: 511},
                              'PubchemFP364': {0: 512, 1: 513}, 'PubchemFP365': {0: 514, 1: 515},
                              'PubchemFP366': {0: 516, 1: 517}, 'PubchemFP367': {0: 518, 1: 519},
                              'PubchemFP368': {0: 520, 1: 521}, 'PubchemFP369': {0: 522},
                              'PubchemFP370': {0: 523, 1: 524}, 'PubchemFP371': {0: 525, 1: 526},
                              'PubchemFP372': {0: 527, 1: 528}, 'PubchemFP373': {0: 529, 1: 530},
                              'PubchemFP374': {0: 531, 1: 532}, 'PubchemFP375': {0: 533, 1: 534},
                              'PubchemFP376': {0: 535, 1: 536}, 'PubchemFP377': {0: 537, 1: 538},
                              'PubchemFP378': {0: 539, 1: 540}, 'PubchemFP379': {0: 541, 1: 542},
                              'PubchemFP380': {0: 543, 1: 544}, 'PubchemFP381': {0: 545, 1: 546},
                              'PubchemFP382': {0: 547, 1: 548}, 'PubchemFP383': {0: 549, 1: 550},
                              'PubchemFP384': {0: 551, 1: 552}, 'PubchemFP385': {0: 553, 1: 554},
                              'PubchemFP386': {0: 555, 1: 556}, 'PubchemFP387': {0: 557, 1: 558},
                              'PubchemFP388': {0: 559, 1: 560}, 'PubchemFP389': {0: 561, 1: 562},
                              'PubchemFP390': {0: 563, 1: 564}, 'PubchemFP391': {0: 565, 1: 566},
                              'PubchemFP392': {0: 567, 1: 568}, 'PubchemFP393': {0: 569, 1: 570},
                              'PubchemFP394': {0: 571, 1: 572}, 'PubchemFP395': {0: 573, 1: 574},
                              'PubchemFP396': {0: 575, 1: 576}, 'PubchemFP397': {0: 577, 1: 578},
                              'PubchemFP398': {0: 579, 1: 580}, 'PubchemFP399': {0: 581, 1: 582},
                              'PubchemFP400': {0: 583, 1: 584}, 'PubchemFP401': {0: 585, 1: 586},
                              'PubchemFP402': {0: 587, 1: 588}, 'PubchemFP403': {0: 589, 1: 590},
                              'PubchemFP404': {0: 591, 1: 592}, 'PubchemFP405': {0: 593, 1: 594},
                              'PubchemFP406': {0: 595, 1: 596}, 'PubchemFP407': {0: 597, 1: 598},
                              'PubchemFP408': {0: 599, 1: 600}, 'PubchemFP409': {0: 601, 1: 602},
                              'PubchemFP410': {0: 603}, 'PubchemFP411': {0: 604, 1: 605},
                              'PubchemFP412': {0: 606, 1: 607}, 'PubchemFP413': {0: 608, 1: 609},
                              'PubchemFP414': {0: 610, 1: 611}, 'PubchemFP415': {0: 612},
                              'PubchemFP416': {0: 613, 1: 614}, 'PubchemFP417': {0: 615, 1: 616},
                              'PubchemFP418': {0: 617, 1: 618}, 'PubchemFP419': {0: 619, 1: 620},
                              'PubchemFP420': {0: 621, 1: 622}, 'PubchemFP421': {0: 623, 1: 624},
                              'PubchemFP422': {0: 625, 1: 626}, 'PubchemFP423': {0: 627, 1: 628},
                              'PubchemFP424': {0: 629}, 'PubchemFP425': {0: 630, 1: 631}, 'PubchemFP426': {0: 632},
                              'PubchemFP427': {0: 633, 1: 634}, 'PubchemFP428': {0: 635, 1: 636},
                              'PubchemFP429': {0: 637, 1: 638}, 'PubchemFP430': {0: 639, 1: 640},
                              'PubchemFP431': {0: 641, 1: 642}, 'PubchemFP432': {0: 643, 1: 644},
                              'PubchemFP433': {0: 645}, 'PubchemFP434': {0: 646, 1: 647},
                              'PubchemFP435': {0: 648, 1: 649}, 'PubchemFP436': {0: 650, 1: 651},
                              'PubchemFP437': {0: 652, 1: 653}, 'PubchemFP438': {0: 654, 1: 655},
                              'PubchemFP439': {0: 656, 1: 657}, 'PubchemFP440': {0: 658, 1: 659},
                              'PubchemFP441': {0: 660, 1: 661}, 'PubchemFP442': {0: 662, 1: 663},
                              'PubchemFP443': {0: 664, 1: 665}, 'PubchemFP444': {0: 666},
                              'PubchemFP445': {0: 667, 1: 668}, 'PubchemFP446': {0: 669, 1: 670},
                              'PubchemFP447': {0: 671, 1: 672}, 'PubchemFP448': {0: 673, 1: 674},
                              'PubchemFP449': {0: 675, 1: 676}, 'PubchemFP450': {0: 677, 1: 678},
                              'PubchemFP451': {0: 679, 1: 680}, 'PubchemFP452': {0: 681, 1: 682},
                              'PubchemFP453': {0: 683, 1: 684}, 'PubchemFP454': {0: 685, 1: 686},
                              'PubchemFP455': {0: 687, 1: 688}, 'PubchemFP456': {0: 689, 1: 690},
                              'PubchemFP457': {0: 691, 1: 692}, 'PubchemFP458': {0: 693, 1: 694},
                              'PubchemFP459': {0: 695, 1: 696}, 'PubchemFP460': {0: 697, 1: 698},
                              'PubchemFP461': {0: 699, 1: 700}, 'PubchemFP462': {0: 701, 1: 702},
                              'PubchemFP463': {0: 703}, 'PubchemFP464': {0: 704, 1: 705},
                              'PubchemFP465': {0: 706, 1: 707}, 'PubchemFP466': {0: 708, 1: 709},
                              'PubchemFP467': {0: 710, 1: 711}, 'PubchemFP468': {0: 712},
                              'PubchemFP469': {0: 713, 1: 714}, 'PubchemFP470': {0: 715, 1: 716},
                              'PubchemFP471': {0: 717, 1: 718}, 'PubchemFP472': {0: 719, 1: 720},
                              'PubchemFP473': {0: 721, 1: 722}, 'PubchemFP474': {0: 723, 1: 724},
                              'PubchemFP475': {0: 725, 1: 726}, 'PubchemFP476': {0: 727, 1: 728},
                              'PubchemFP477': {0: 729, 1: 730}, 'PubchemFP478': {0: 731},
                              'PubchemFP479': {0: 732, 1: 733}, 'PubchemFP480': {0: 734, 1: 735},
                              'PubchemFP481': {0: 736, 1: 737}, 'PubchemFP482': {0: 738, 1: 739},
                              'PubchemFP483': {0: 740, 1: 741}, 'PubchemFP484': {0: 742, 1: 743},
                              'PubchemFP485': {0: 744, 1: 745}, 'PubchemFP486': {0: 746, 1: 747},
                              'PubchemFP487': {0: 748, 1: 749}, 'PubchemFP488': {0: 750, 1: 751},
                              'PubchemFP489': {0: 752, 1: 753}, 'PubchemFP490': {0: 754, 1: 755},
                              'PubchemFP491': {0: 756, 1: 757}, 'PubchemFP492': {0: 758, 1: 759},
                              'PubchemFP493': {0: 760, 1: 761}, 'PubchemFP494': {0: 762, 1: 763},
                              'PubchemFP495': {0: 764, 1: 765}, 'PubchemFP496': {0: 766, 1: 767},
                              'PubchemFP497': {0: 768, 1: 769}, 'PubchemFP498': {0: 770, 1: 771},
                              'PubchemFP499': {0: 772, 1: 773}, 'PubchemFP500': {0: 774, 1: 775},
                              'PubchemFP501': {0: 776, 1: 777}, 'PubchemFP502': {0: 778, 1: 779},
                              'PubchemFP503': {0: 780, 1: 781}, 'PubchemFP504': {0: 782, 1: 783},
                              'PubchemFP505': {0: 784, 1: 785}, 'PubchemFP506': {0: 786, 1: 787},
                              'PubchemFP507': {0: 788, 1: 789}, 'PubchemFP508': {0: 790, 1: 791},
                              'PubchemFP509': {0: 792, 1: 793}, 'PubchemFP510': {0: 794, 1: 795},
                              'PubchemFP511': {0: 796, 1: 797}, 'PubchemFP512': {0: 798},
                              'PubchemFP513': {0: 799, 1: 800}, 'PubchemFP514': {0: 801, 1: 802},
                              'PubchemFP515': {0: 803, 1: 804}, 'PubchemFP516': {0: 805, 1: 806},
                              'PubchemFP517': {0: 807, 1: 808}, 'PubchemFP518': {0: 809, 1: 810},
                              'PubchemFP519': {0: 811, 1: 812}, 'PubchemFP520': {0: 813, 1: 814},
                              'PubchemFP521': {0: 815, 1: 816}, 'PubchemFP522': {0: 817, 1: 818},
                              'PubchemFP523': {0: 819, 1: 820}, 'PubchemFP524': {0: 821, 1: 822},
                              'PubchemFP525': {0: 823}, 'PubchemFP526': {0: 824}, 'PubchemFP527': {0: 825, 1: 826},
                              'PubchemFP528': {0: 827, 1: 828}, 'PubchemFP529': {0: 829},
                              'PubchemFP530': {0: 830, 1: 831}, 'PubchemFP531': {0: 832, 1: 833},
                              'PubchemFP532': {0: 834, 1: 835}, 'PubchemFP533': {0: 836, 1: 837},
                              'PubchemFP534': {0: 838, 1: 839}, 'PubchemFP535': {0: 840, 1: 841},
                              'PubchemFP536': {0: 842, 1: 843}, 'PubchemFP537': {0: 844, 1: 845},
                              'PubchemFP538': {0: 846, 1: 847}, 'PubchemFP539': {0: 848, 1: 849},
                              'PubchemFP540': {0: 850, 1: 851}, 'PubchemFP541': {0: 852, 1: 853},
                              'PubchemFP542': {0: 854, 1: 855}, 'PubchemFP543': {0: 856, 1: 857},
                              'PubchemFP544': {0: 858, 1: 859}, 'PubchemFP545': {0: 860, 1: 861},
                              'PubchemFP546': {0: 862, 1: 863}, 'PubchemFP547': {0: 864, 1: 865},
                              'PubchemFP548': {0: 866, 1: 867}, 'PubchemFP549': {0: 868, 1: 869},
                              'PubchemFP550': {0: 870, 1: 871}, 'PubchemFP551': {0: 872, 1: 873},
                              'PubchemFP552': {0: 874, 1: 875}, 'PubchemFP553': {0: 876, 1: 877},
                              'PubchemFP554': {0: 878, 1: 879}, 'PubchemFP555': {0: 880, 1: 881},
                              'PubchemFP556': {0: 882, 1: 883}, 'PubchemFP557': {0: 884},
                              'PubchemFP558': {0: 885, 1: 886}, 'PubchemFP559': {0: 887, 1: 888},
                              'PubchemFP560': {0: 889, 1: 890}, 'PubchemFP561': {0: 891, 1: 892},
                              'PubchemFP562': {0: 893}, 'PubchemFP563': {0: 894, 1: 895},
                              'PubchemFP564': {0: 896, 1: 897}, 'PubchemFP565': {0: 898, 1: 899},
                              'PubchemFP566': {0: 900, 1: 901}, 'PubchemFP567': {0: 902, 1: 903},
                              'PubchemFP568': {0: 904, 1: 905}, 'PubchemFP569': {0: 906, 1: 907},
                              'PubchemFP570': {0: 908, 1: 909}, 'PubchemFP571': {0: 910, 1: 911},
                              'PubchemFP572': {0: 912, 1: 913}, 'PubchemFP573': {0: 914, 1: 915},
                              'PubchemFP574': {0: 916, 1: 917}, 'PubchemFP575': {0: 918, 1: 919},
                              'PubchemFP576': {0: 920, 1: 921}, 'PubchemFP577': {0: 922, 1: 923},
                              'PubchemFP578': {0: 924, 1: 925}, 'PubchemFP579': {0: 926, 1: 927},
                              'PubchemFP580': {0: 928, 1: 929}, 'PubchemFP581': {0: 930, 1: 931},
                              'PubchemFP582': {0: 932, 1: 933}, 'PubchemFP583': {0: 934, 1: 935},
                              'PubchemFP584': {0: 936, 1: 937}, 'PubchemFP585': {0: 938, 1: 939},
                              'PubchemFP586': {0: 940, 1: 941}, 'PubchemFP587': {0: 942},
                              'PubchemFP588': {0: 943, 1: 944}, 'PubchemFP589': {0: 945, 1: 946},
                              'PubchemFP590': {0: 947, 1: 948}, 'PubchemFP591': {0: 949, 1: 950},
                              'PubchemFP592': {0: 951, 1: 952}, 'PubchemFP593': {0: 953, 1: 954},
                              'PubchemFP594': {0: 955, 1: 956}, 'PubchemFP595': {0: 957, 1: 958},
                              'PubchemFP596': {0: 959, 1: 960}, 'PubchemFP597': {0: 961, 1: 962},
                              'PubchemFP598': {0: 963, 1: 964}, 'PubchemFP599': {0: 965, 1: 966},
                              'PubchemFP600': {0: 967, 1: 968}, 'PubchemFP601': {0: 969, 1: 970},
                              'PubchemFP602': {0: 971, 1: 972}, 'PubchemFP603': {0: 973, 1: 974},
                              'PubchemFP604': {0: 975, 1: 976}, 'PubchemFP605': {0: 977, 1: 978},
                              'PubchemFP606': {0: 979, 1: 980}, 'PubchemFP607': {0: 981, 1: 982},
                              'PubchemFP608': {0: 983, 1: 984}, 'PubchemFP609': {0: 985, 1: 986},
                              'PubchemFP610': {0: 987, 1: 988}, 'PubchemFP611': {0: 989, 1: 990},
                              'PubchemFP612': {0: 991, 1: 992}, 'PubchemFP613': {0: 993, 1: 994},
                              'PubchemFP614': {0: 995, 1: 996}, 'PubchemFP615': {0: 997, 1: 998},
                              'PubchemFP616': {0: 999, 1: 1000}, 'PubchemFP617': {0: 1001, 1: 1002},
                              'PubchemFP618': {0: 1003, 1: 1004}, 'PubchemFP619': {0: 1005, 1: 1006},
                              'PubchemFP620': {0: 1007, 1: 1008}, 'PubchemFP621': {0: 1009, 1: 1010},
                              'PubchemFP622': {0: 1011, 1: 1012}, 'PubchemFP623': {0: 1013, 1: 1014},
                              'PubchemFP624': {0: 1015, 1: 1016}, 'PubchemFP625': {0: 1017, 1: 1018},
                              'PubchemFP626': {0: 1019, 1: 1020}, 'PubchemFP627': {0: 1021},
                              'PubchemFP628': {0: 1022, 1: 1023}, 'PubchemFP629': {0: 1024, 1: 1025},
                              'PubchemFP630': {0: 1026, 1: 1027}, 'PubchemFP631': {0: 1028, 1: 1029},
                              'PubchemFP632': {0: 1030, 1: 1031}, 'PubchemFP633': {0: 1032, 1: 1033},
                              'PubchemFP634': {0: 1034, 1: 1035}, 'PubchemFP635': {0: 1036, 1: 1037},
                              'PubchemFP636': {0: 1038, 1: 1039}, 'PubchemFP637': {0: 1040, 1: 1041},
                              'PubchemFP638': {0: 1042, 1: 1043}, 'PubchemFP639': {0: 1044, 1: 1045},
                              'PubchemFP640': {0: 1046, 1: 1047}, 'PubchemFP641': {0: 1048, 1: 1049},
                              'PubchemFP642': {0: 1050, 1: 1051}, 'PubchemFP643': {0: 1052, 1: 1053},
                              'PubchemFP644': {0: 1054, 1: 1055}, 'PubchemFP645': {0: 1056, 1: 1057},
                              'PubchemFP646': {0: 1058, 1: 1059}, 'PubchemFP647': {0: 1060, 1: 1061},
                              'PubchemFP648': {0: 1062, 1: 1063}, 'PubchemFP649': {0: 1064},
                              'PubchemFP650': {0: 1065, 1: 1066}, 'PubchemFP651': {0: 1067, 1: 1068},
                              'PubchemFP652': {0: 1069, 1: 1070}, 'PubchemFP653': {0: 1071, 1: 1072},
                              'PubchemFP654': {0: 1073, 1: 1074}, 'PubchemFP655': {0: 1075, 1: 1076},
                              'PubchemFP656': {0: 1077, 1: 1078}, 'PubchemFP657': {0: 1079, 1: 1080},
                              'PubchemFP658': {0: 1081, 1: 1082}, 'PubchemFP659': {0: 1083, 1: 1084},
                              'PubchemFP660': {0: 1085, 1: 1086}, 'PubchemFP661': {0: 1087, 1: 1088},
                              'PubchemFP662': {0: 1089, 1: 1090}, 'PubchemFP663': {0: 1091, 1: 1092},
                              'PubchemFP664': {0: 1093, 1: 1094}, 'PubchemFP665': {0: 1095, 1: 1096},
                              'PubchemFP666': {0: 1097, 1: 1098}, 'PubchemFP667': {0: 1099, 1: 1100},
                              'PubchemFP668': {0: 1101, 1: 1102}, 'PubchemFP669': {0: 1103, 1: 1104},
                              'PubchemFP670': {0: 1105, 1: 1106}, 'PubchemFP671': {0: 1107, 1: 1108},
                              'PubchemFP672': {0: 1109, 1: 1110}, 'PubchemFP673': {0: 1111, 1: 1112},
                              'PubchemFP674': {0: 1113, 1: 1114}, 'PubchemFP675': {0: 1115, 1: 1116},
                              'PubchemFP676': {0: 1117, 1: 1118}, 'PubchemFP677': {0: 1119, 1: 1120},
                              'PubchemFP678': {0: 1121, 1: 1122}, 'PubchemFP679': {0: 1123, 1: 1124},
                              'PubchemFP680': {0: 1125, 1: 1126}, 'PubchemFP681': {0: 1127, 1: 1128},
                              'PubchemFP682': {0: 1129, 1: 1130}, 'PubchemFP683': {0: 1131, 1: 1132},
                              'PubchemFP684': {0: 1133, 1: 1134}, 'PubchemFP685': {0: 1135, 1: 1136},
                              'PubchemFP686': {0: 1137, 1: 1138}, 'PubchemFP687': {0: 1139, 1: 1140},
                              'PubchemFP688': {0: 1141, 1: 1142}, 'PubchemFP689': {0: 1143, 1: 1144},
                              'PubchemFP690': {0: 1145, 1: 1146}, 'PubchemFP691': {0: 1147, 1: 1148},
                              'PubchemFP692': {0: 1149, 1: 1150}, 'PubchemFP693': {0: 1151, 1: 1152},
                              'PubchemFP694': {0: 1153, 1: 1154}, 'PubchemFP695': {0: 1155, 1: 1156},
                              'PubchemFP696': {0: 1157, 1: 1158}, 'PubchemFP697': {0: 1159, 1: 1160},
                              'PubchemFP698': {0: 1161, 1: 1162}, 'PubchemFP699': {0: 1163, 1: 1164},
                              'PubchemFP700': {0: 1165, 1: 1166}, 'PubchemFP701': {0: 1167, 1: 1168},
                              'PubchemFP702': {0: 1169, 1: 1170}, 'PubchemFP703': {0: 1171, 1: 1172},
                              'PubchemFP704': {0: 1173, 1: 1174}, 'PubchemFP705': {0: 1175, 1: 1176},
                              'PubchemFP706': {0: 1177, 1: 1178}, 'PubchemFP707': {0: 1179, 1: 1180},
                              'PubchemFP708': {0: 1181, 1: 1182}, 'PubchemFP709': {0: 1183, 1: 1184},
                              'PubchemFP710': {0: 1185, 1: 1186}, 'PubchemFP711': {0: 1187, 1: 1188},
                              'PubchemFP712': {0: 1189, 1: 1190}, 'PubchemFP713': {0: 1191, 1: 1192},
                              'PubchemFP714': {0: 1193, 1: 1194}, 'PubchemFP715': {0: 1195, 1: 1196},
                              'PubchemFP716': {0: 1197, 1: 1198}, 'PubchemFP717': {0: 1199, 1: 1200},
                              'PubchemFP718': {0: 1201}, 'PubchemFP719': {0: 1202, 1: 1203},
                              'PubchemFP720': {0: 1204, 1: 1205}, 'PubchemFP721': {0: 1206, 1: 1207},
                              'PubchemFP722': {0: 1208, 1: 1209}, 'PubchemFP723': {0: 1210}, 'PubchemFP724': {0: 1211},
                              'PubchemFP725': {0: 1212, 1: 1213}, 'PubchemFP726': {0: 1214, 1: 1215},
                              'PubchemFP727': {0: 1216}, 'PubchemFP728': {0: 1217, 1: 1218},
                              'PubchemFP729': {0: 1219, 1: 1220}, 'PubchemFP730': {0: 1221, 1: 1222},
                              'PubchemFP731': {0: 1223, 1: 1224}, 'PubchemFP732': {0: 1225},
                              'PubchemFP733': {0: 1226, 1: 1227}, 'PubchemFP734': {0: 1228, 1: 1229},
                              'PubchemFP735': {0: 1230, 1: 1231}, 'PubchemFP736': {0: 1232, 1: 1233},
                              'PubchemFP737': {0: 1234, 1: 1235}, 'PubchemFP738': {0: 1236, 1: 1237},
                              'PubchemFP739': {0: 1238}, 'PubchemFP740': {0: 1239, 1: 1240},
                              'PubchemFP741': {0: 1241, 1: 1242}, 'PubchemFP742': {0: 1243, 1: 1244},
                              'PubchemFP743': {0: 1245, 1: 1246}, 'PubchemFP744': {0: 1247},
                              'PubchemFP745': {0: 1248, 1: 1249}, 'PubchemFP746': {0: 1250, 1: 1251},
                              'PubchemFP747': {0: 1252, 1: 1253}, 'PubchemFP748': {0: 1254},
                              'PubchemFP749': {0: 1255, 1: 1256}, 'PubchemFP750': {0: 1257, 1: 1258},
                              'PubchemFP751': {0: 1259, 1: 1260}, 'PubchemFP752': {0: 1261, 1: 1262},
                              'PubchemFP753': {0: 1263, 1: 1264}, 'PubchemFP754': {0: 1265, 1: 1266},
                              'PubchemFP755': {0: 1267, 1: 1268}, 'PubchemFP756': {0: 1269, 1: 1270},
                              'PubchemFP757': {0: 1271, 1: 1272}, 'PubchemFP758': {0: 1273, 1: 1274},
                              'PubchemFP759': {0: 1275, 1: 1276}, 'PubchemFP760': {0: 1277},
                              'PubchemFP761': {0: 1278, 1: 1279}, 'PubchemFP762': {0: 1280, 1: 1281},
                              'PubchemFP763': {0: 1282, 1: 1283}, 'PubchemFP764': {0: 1284, 1: 1285},
                              'PubchemFP765': {0: 1286, 1: 1287}, 'PubchemFP766': {0: 1288, 1: 1289},
                              'PubchemFP767': {0: 1290, 1: 1291}, 'PubchemFP768': {0: 1292, 1: 1293},
                              'PubchemFP769': {0: 1294}, 'PubchemFP770': {0: 1295, 1: 1296},
                              'PubchemFP771': {0: 1297, 1: 1298}, 'PubchemFP772': {0: 1299, 1: 1300},
                              'PubchemFP773': {0: 1301}, 'PubchemFP774': {0: 1302}, 'PubchemFP775': {0: 1303},
                              'PubchemFP776': {0: 1304, 1: 1305}, 'PubchemFP777': {0: 1306, 1: 1307},
                              'PubchemFP778': {0: 1308, 1: 1309}, 'PubchemFP779': {0: 1310, 1: 1311},
                              'PubchemFP780': {0: 1312, 1: 1313}, 'PubchemFP781': {0: 1314},
                              'PubchemFP782': {0: 1315, 1: 1316}, 'PubchemFP783': {0: 1317, 1: 1318},
                              'PubchemFP784': {0: 1319, 1: 1320}, 'PubchemFP785': {0: 1321, 1: 1322},
                              'PubchemFP786': {0: 1323}, 'PubchemFP787': {0: 1324}, 'PubchemFP788': {0: 1325, 1: 1326},
                              'PubchemFP789': {0: 1327, 1: 1328}, 'PubchemFP790': {0: 1329},
                              'PubchemFP791': {0: 1330, 1: 1331}, 'PubchemFP792': {0: 1332, 1: 1333},
                              'PubchemFP793': {0: 1334, 1: 1335}, 'PubchemFP794': {0: 1336, 1: 1337},
                              'PubchemFP795': {0: 1338}, 'PubchemFP796': {0: 1339, 1: 1340},
                              'PubchemFP797': {0: 1341, 1: 1342}, 'PubchemFP798': {0: 1343, 1: 1344},
                              'PubchemFP799': {0: 1345, 1: 1346}, 'PubchemFP800': {0: 1347, 1: 1348},
                              'PubchemFP801': {0: 1349, 1: 1350}, 'PubchemFP802': {0: 1351},
                              'PubchemFP803': {0: 1352, 1: 1353}, 'PubchemFP804': {0: 1354, 1: 1355},
                              'PubchemFP805': {0: 1356, 1: 1357}, 'PubchemFP806': {0: 1358, 1: 1359},
                              'PubchemFP807': {0: 1360}, 'PubchemFP808': {0: 1361, 1: 1362},
                              'PubchemFP809': {0: 1363, 1: 1364}, 'PubchemFP810': {0: 1365, 1: 1366},
                              'PubchemFP811': {0: 1367}, 'PubchemFP812': {0: 1368, 1: 1369},
                              'PubchemFP813': {0: 1370, 1: 1371}, 'PubchemFP814': {0: 1372, 1: 1373},
                              'PubchemFP815': {0: 1374, 1: 1375}, 'PubchemFP816': {0: 1376, 1: 1377},
                              'PubchemFP817': {0: 1378, 1: 1379}, 'PubchemFP818': {0: 1380, 1: 1381},
                              'PubchemFP819': {0: 1382, 1: 1383}, 'PubchemFP820': {0: 1384, 1: 1385},
                              'PubchemFP821': {0: 1386, 1: 1387}, 'PubchemFP822': {0: 1388, 1: 1389},
                              'PubchemFP823': {0: 1390}, 'PubchemFP824': {0: 1391, 1: 1392},
                              'PubchemFP825': {0: 1393, 1: 1394}, 'PubchemFP826': {0: 1395, 1: 1396},
                              'PubchemFP827': {0: 1397, 1: 1398}, 'PubchemFP828': {0: 1399, 1: 1400},
                              'PubchemFP829': {0: 1401, 1: 1402}, 'PubchemFP830': {0: 1403, 1: 1404},
                              'PubchemFP831': {0: 1405, 1: 1406}, 'PubchemFP832': {0: 1407},
                              'PubchemFP833': {0: 1408, 1: 1409}, 'PubchemFP834': {0: 1410, 1: 1411},
                              'PubchemFP835': {0: 1412, 1: 1413}, 'PubchemFP836': {0: 1414}, 'PubchemFP837': {0: 1415},
                              'PubchemFP838': {0: 1416}, 'PubchemFP839': {0: 1417, 1: 1418},
                              'PubchemFP840': {0: 1419, 1: 1420}, 'PubchemFP841': {0: 1421},
                              'PubchemFP842': {0: 1422, 1: 1423}, 'PubchemFP843': {0: 1424}, 'PubchemFP844': {0: 1425},
                              'PubchemFP845': {0: 1426}, 'PubchemFP846': {0: 1427}, 'PubchemFP847': {0: 1428, 1: 1429},
                              'PubchemFP848': {0: 1430}, 'PubchemFP849': {0: 1431}, 'PubchemFP850': {0: 1432},
                              'PubchemFP851': {0: 1433}, 'PubchemFP852': {0: 1434}, 'PubchemFP853': {0: 1435},
                              'PubchemFP854': {0: 1436}, 'PubchemFP855': {0: 1437}, 'PubchemFP856': {0: 1438},
                              'PubchemFP857': {0: 1439}, 'PubchemFP858': {0: 1440}, 'PubchemFP859': {0: 1441},
                              'PubchemFP860': {0: 1442, 1: 1443}, 'PubchemFP861': {0: 1444, 1: 1445},
                              'PubchemFP862': {0: 1446}, 'PubchemFP863': {0: 1447, 1: 1448}, 'PubchemFP864': {0: 1449},
                              'PubchemFP865': {0: 1450}, 'PubchemFP866': {0: 1451}, 'PubchemFP867': {0: 1452},
                              'PubchemFP868': {0: 1453}, 'PubchemFP869': {0: 1454}, 'PubchemFP870': {0: 1455},
                              'PubchemFP871': {0: 1456}, 'PubchemFP872': {0: 1457}, 'PubchemFP873': {0: 1458},
                              'PubchemFP874': {0: 1459}, 'PubchemFP875': {0: 1460}, 'PubchemFP876': {0: 1461},
                              'PubchemFP877': {0: 1462}, 'PubchemFP878': {0: 1463}, 'PubchemFP879': {0: 1464},
                              'PubchemFP880': {0: 1465}}

            self.feat_dim = 1466




    def gen_feat_dict(self):

        cate_df = self.catefile

        self.feat_dict = {}

        self.feat_len = {}

        tc = 0


        # 当数据集很小时，可以这样，当数据集很大时，就不方便了
        for col in cate_df.columns:

            if col in self.cate_cols:
                # print("col:")
                # print(col)

                us = cate_df[col].unique()

                self.feat_dict[col] = dict(zip(us, range(tc, len(us) + tc)))

                tc += len(us)

        self.feat_dim = tc







class DataParser(object):

    def __init__(self, feat_dict):

        self.feat_dict = feat_dict





    def parse(self, infile=None, df=None, has_label=False):

        assert not ((infile is None) and (df is None)), "infile or df at least one is set"

        assert not ((infile is not None) and (df is not None)), "only one can be set"

        if infile is not None:

            df = pd.read_csv(infile)

        if has_label:

            y_cols = ['c_indices', 'd_indices', 'target']

            y = df[y_cols].values.tolist()

            # y_cols = ['target']
            #
            # y = dfi[y_cols].values.tolist()

            id_cols = ['c_indices', 'd_indices', 'id']

            ids = df[id_cols].values.tolist()

            df.drop(["id", "target", "c_indices", "d_indices"], axis=1, inplace=True)

        else:

            id_cols = ['c_indices', 'd_indices', 'id']

            ids = df[id_cols].values.tolist()

            df.drop(["id", "target", "c_indices", "d_indices"], axis=1, inplace=True)

        # dfi for feature index

        # dfv for feature value which can be either binary (1/0) or float (e.g., 10.24)



        numeric_Xv = df[self.feat_dict.numeric_cols].values.tolist()

        df.drop(self.feat_dict.numeric_cols, axis=1, inplace=True)




        dfv = df.copy()

        for col in df.columns:

            if col in self.feat_dict.ignore_cols:

                df.drop(col, axis=1, inplace=True)

                dfv.drop(col, axis=1, inplace=True)

                continue

            else:

                # print("dfi[col]")
                # print(dfi[col])

                df[col] = df[col].map(self.feat_dict.feat_dict[col])

                dfv[col] = 1.



        # list of list of feature indices of each sample in the dataset

        cate_Xi = df.values.tolist()

        # list of list of feature values of each sample in the dataset

        cate_Xv = dfv.values.tolist()

        del df
        gc.collect()

        del dfv
        gc.collect()

        if has_label:

            return cate_Xi, cate_Xv, numeric_Xv, y, ids

        else:

            return cate_Xi, cate_Xv, numeric_Xv, ids



def gen_pairs( y, ids):
    # print("gen_pairs start: *******************")

    pairs = []

    for i in range(0, len(ids)):

        for j in range(i + 1, len(ids)):
            # Only look at queries with the same id
            # print(ids[j])
            if (ids[i][0] != ids[j][0]):
                continue
            # Document pairs found with different rating
            if (ids[i][0] == ids[j][0] and y[i] != y[j]): ## 相同cell-id，不同的drug_id
                # Sort by saving the largest index in position 0
                # print("y[i]:")
                # print(y[i])
                if (y[i] > y[j]):
                    pairs.append([ids[i][0], int(i), int(j)]) #ids[i][0] 是cell-id  i,j 是针对 ids的序号，

                else:
                    pairs.append([ids[i][0],int(j), int(i)])

    return pairs



def gen_pairs_( y_id):
    # print("gen_pairs start: *******************")

    pairs = []

    for i in range(0, len(y_id)):
        # print("pair****:")
        # print(ids[i])
        for j in range(i + 1, len(y_id)):
            # Only look at queries with the same id
            # print(ids[j])
            if (y_id[i][0] != y_id[j][0]):
                continue
            # Document pairs found with different rating
            if (y_id[i][0] == y_id[j][0] and y_id[i][1] != y_id[j][1]): ## 相同cell-id，不同的drug_id
                # Sort by saving the largest index in position 0
                # print("y[i]:")
                # print("y_id[i][2] : %f, y_id[j][2] : %f" %y_id[i][2]%y_id[j][2])
                if (y_id[i][2] > y_id[j][2]):
                    pairs.append([y_id[i][0], int(y_id[i][3]), int(y_id[j][3])]) #ids[i][0] 是cell-id  i,j 是针对 ids的序号，

                else:
                    pairs.append([y_id[i][0],int(y_id[j][3]), int(y_id[i][3])])

    return pairs


def gen_pairs_count_( y_id):

    count = 0
    for i in range(0, len(y_id)):
        # print("pair****:")
        # print(ids[i])
        for j in range(i + 1, len(y_id)):
            # Only look at queries with the same id
            # print(ids[j])
            if (y_id[i][0] != y_id[j][0]):
                continue
            # Document pairs found with different rating
            if (y_id[i][0] == y_id[j][0] and y_id[i][1] != y_id[j][1]): ## 相同cell-id，不同的drug_id
                # Sort by saving the largest index in position 0
                # print("y[i]:")
                print("y_id[i][2] : %f, y_id[j][2] : %f" ,[y_id[i][2], y_id[j][2]])
                count += 1
    #
    return count

def gen_pairs_count( y, ids, show = None):
    # print("gen_pairs start: *******************")


    count = 0

    for i in range(0, len(ids)):
        # print("pair****:")
        # print(ids[i])
        for j in range(i + 1, len(ids)):
            # Only look at queries with the same id
            # print(ids[j])

            if (ids[i][0] != ids[j][0]):
                if show:
                    print("ids[i][0]: %s -- ids[j][0]: %s : can be pair" % str(ids[i][0]) % str(ids[j][0]))

                continue
            # Document pairs found with different rating
            if (ids[i][0] == ids[j][0] and y[i] != y[j]): ## 相同cell-id，不同的drug_id
                # Sort by saving the largest index in position 0
                # print("y[i]:")
                # print(y[i])
                count += 1

    return count


def gen_pairs_by_cell_line(train_array):
    pairs = []

    y_id_train_df = pd.DataFrame(train_array, columns=config.cols)
    c_train = np.unique(y_id_train_df['c_indices'])
    print(len(c_train))
    K = config.K  ##每个drug被取到15次

    for each in c_train:
        # print("c_indices: %d"%each)
        each_df = y_id_train_df[y_id_train_df['c_indices'] == each]
        each_array = np.array(each_df)
        # print(each_array)

        drugs_index = []
        for i in range(0,len(each_df)-1):
            drugs_index.append(i)

        d_dict = {}
        d_count = {}
        undo_drug = []

        for i in range(0, len(each_array)-1):
            each_drug = int(each_array[i][1])
            # print(each_drug)
            selective_drugs_index = drugs_index[:]

            count = 0
            selective_drugs_index.remove(i)
            # selective_drugs_index = random.sample(selective_drugs_index, min(K,len(selective_drugs_index)))
            # for j in selective_drugs_index:
            while count < K:
                if len(selective_drugs_index) == 0:
                    break

                j = random.sample(selective_drugs_index, 1)[0]
                if j > i:


                    if j not in undo_drug:

                        if (each_array[i][2] > each_array[j][2]):
                            pairs.append([each_array[i][0], int(each_array[i][3]),
                                          int(each_array[j][3])])
                        else:
                            pairs.append([each_array[i][0], int(each_array[j][3]), int(each_array[i][3])])
                        if i not in d_dict.keys():
                            d_dict[i] = [j]
                        else:
                            temp = d_dict[i]
                            temp.append(j)
                            d_dict[i] = temp
                        count += 1

                        ### 计入d_count
                        if j not in d_count.keys():
                            d_count[j] = 1
                        else:
                            d_count[j] += 1
                            if d_count[j] == len(drugs_index) - K-1:
                                undo_drug.append(j)
                    else:
                        selective_drugs_index.remove(j)
                        continue

                else:

                    use_index = d_dict[j]
                    if i in use_index:
                        selective_drugs_index.remove(j)
                        continue
                    else:
                        if (each_array[i][2] > each_array[j][2]):
                            pairs.append([each_array[i][0], int(each_array[i][3]),
                                          int(each_array[j][3])])  # ids[i][0] 是cell-id  i,j 是针对 ids的序号，
                        else:
                            pairs.append([each_array[i][0], int(each_array[j][3]), int(each_array[i][3])])

                        if i not in d_dict.keys():
                            d_dict[i] = [j]
                        else:
                            temp = d_dict[i]
                            temp.append(j)
                            d_dict[i] = temp
                        count += 1

                selective_drugs_index.remove(j)

                # print(len(selective_drugs_index))

    return pairs




class df_Resolve_To_numpy(object):
    def __init__(self, df, data_parser):
        # self.df = df
        self.cate_Xi,self.cate_Xv,self.numeric_Xv, self.y,self.id = data_parser.parse(df=df, has_label=True)


    def get_numpy(self):
        return self.cate_Xi,self.cate_Xv,self.numeric_Xv, self.y,self.id

    def get_cate_Xi(self):
        return self.cate_Xi

    def get_cate_Xv(self):
        return self.cate_Xv

    def get_numeric_Xv(self):
        return self.numeric_Xv

    def get_y(self):
        return self.y

    def get_id(self):
        return self.id
