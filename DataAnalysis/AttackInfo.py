'''
攻擊事件清單

來源：List of Attacks Final.pdf — SUTD SWaT 實驗室官方攻擊紀錄

攻擊分類：
- SSSP = Single Stage Single Point
- SSMP = Single Stage Multi Point
- MSMP = Multi Stage Multi Point
- NPI  = No Physical Impact(無實際物理影響)

Actual Change = 攻擊是否確實改變了物理設備狀態(Yes=執行器攻擊, No=感測器欺騙）
'''

attack_info = {
    1:  {'point': ['MV101'],                  'stage': [1],    'type': 'actuator', 'actual_change': True,  'impact': True,  'desc': 'Open MV-101',                                       'intent': 'Tank overflow',                    'cat': 'SSSP'},
    2:  {'point': ['P102'],                   'stage': [1],    'type': 'actuator', 'actual_change': True,  'impact': True,  'desc': 'Turn on P-102',                                     'intent': 'Pipe bursts',                      'cat': 'SSSP'},
    3:  {'point': ['LIT101'],                 'stage': [1],    'type': 'sensor',   'actual_change': False, 'impact': True,  'desc': 'Spoof LIT-101 +1mm/s',                              'intent': 'Tank underflow; Damage P-101',     'cat': 'SSSP'},
    4:  {'point': ['MV504'],                  'stage': [5],    'type': 'actuator', 'actual_change': True,  'impact': False, 'desc': 'Open MV-504',                                       'intent': 'Halt RO shutdown',                 'cat': 'SSSP'},
    5:  {'point': [],                         'stage': [],     'type': 'none',     'actual_change': False, 'impact': False, 'desc': 'No Physical Impact',                                'intent': 'N/A',                              'cat': 'NPI'},
    6:  {'point': ['AIT202'],                 'stage': [2],    'type': 'sensor',   'actual_change': False, 'impact': True,  'desc': 'Set AIT-202 = 6',                                   'intent': 'P-203 off; Water quality change',  'cat': 'SSSP'},
    7:  {'point': ['LIT301'],                 'stage': [3],    'type': 'sensor',   'actual_change': False, 'impact': True,  'desc': 'Spoof LIT-301 > HH',                                'intent': 'Stop inflow; Tank underflow',      'cat': 'SSSP'},
    8:  {'point': ['DPIT301'],                'stage': [3],    'type': 'sensor',   'actual_change': False, 'impact': True,  'desc': 'Set DPIT-301 > 40kPa',                              'intent': 'Backwash restart loop',            'cat': 'SSSP'},
    9:  {'point': [],                         'stage': [],     'type': 'none',     'actual_change': False, 'impact': False, 'desc': 'No Physical Impact',                                'intent': 'N/A',                              'cat': 'NPI'},
    10: {'point': ['FIT401'],                 'stage': [4],    'type': 'sensor',   'actual_change': False, 'impact': True,  'desc': 'Set FIT-401 < 0.7',                                 'intent': 'UV shutdown; P-501 off',           'cat': 'SSSP'},
    11: {'point': ['FIT401'],                 'stage': [4],    'type': 'sensor',   'actual_change': False, 'impact': True,  'desc': 'Set FIT-401 = 0',                                   'intent': 'UV shutdown; P-501 off',           'cat': 'SSSP'},
    12: {'point': [],                         'stage': [],     'type': 'none',     'actual_change': False, 'impact': False, 'desc': 'No Physical Impact',                                'intent': 'N/A',                              'cat': 'NPI'},
    13: {'point': ['MV304'],                  'stage': [3],    'type': 'actuator', 'actual_change': True,  'impact': True,  'desc': 'Close MV-304',                                      'intent': 'Halt stage-3 backwash',            'cat': 'SSSP'},
    14: {'point': ['MV303'],                  'stage': [3],    'type': 'actuator', 'actual_change': True,  'impact': True,  'desc': 'Keep MV-303 closed',                                'intent': 'Halt stage-3 backwash',            'cat': 'SSSP'},
    15: {'point': [],                         'stage': [],     'type': 'none',     'actual_change': False, 'impact': False, 'desc': 'No Physical Impact',                                'intent': 'N/A',                              'cat': 'NPI'},
    16: {'point': ['LIT301'],                 'stage': [3],    'type': 'sensor',   'actual_change': False, 'impact': True,  'desc': 'Spoof LIT-301 -1mm/s',                              'intent': 'Tank overflow',                    'cat': 'SSSP'},
    17: {'point': ['MV303'],                  'stage': [3],    'type': 'actuator', 'actual_change': True,  'impact': True,  'desc': 'Keep MV-303 closed',                                'intent': 'Halt stage-3 backwash',            'cat': 'SSSP'},
    18: {'point': [],                         'stage': [],     'type': 'none',     'actual_change': False, 'impact': False, 'desc': 'No Physical Impact',                                'intent': 'N/A',                              'cat': 'NPI'},
    19: {'point': ['AIT504'],                 'stage': [5],    'type': 'sensor',   'actual_change': False, 'impact': False, 'desc': 'Set AIT-504 = 16 uS/cm',                            'intent': 'RO shutdown sequence',             'cat': 'SSSP'},
    20: {'point': ['AIT504'],                 'stage': [5],    'type': 'sensor',   'actual_change': False, 'impact': False, 'desc': 'Set AIT-504 = 255 uS/cm',                           'intent': 'RO shutdown sequence',             'cat': 'SSSP'},
    21: {'point': ['MV101', 'LIT101'],        'stage': [1],    'type': 'mixed',    'actual_change': True,  'impact': True,  'desc': 'Keep MV-101 on; Spoof LIT-101',                     'intent': 'Tank overflow',                    'cat': 'SSMP'},
    22: {'point': ['UV401','AIT502','P501'],  'stage': [4,5],  'type': 'mixed',    'actual_change': True,  'impact': True,  'desc': 'Stop UV-401; Set AIT-502=150; Close P-501',         'intent': 'Damage to RO',                     'cat': 'MSMP'},
    23: {'point': ['P602','DPIT301','MV302'], 'stage': [3,6],  'type': 'mixed',   'actual_change': True,   'impact': True,  'desc': 'Set DPIT-301>0.4bar; Keep MV-302; Change P-602',    'intent': 'System freeze',                    'cat': 'MSMP'},
    24: {'point': ['P203','P205'],            'stage': [2],    'type': 'actuator', 'actual_change': True,  'impact': True,  'desc': 'Turn off P-203 and P-205',                          'intent': 'Water quality change',             'cat': 'SSMP'},
    25: {'point': ['LIT401','P401'],          'stage': [4],    'type': 'mixed',    'actual_change': True,  'impact': True,  'desc': 'Set LIT-401=1000; Keep P-402 on',                   'intent': 'Tank underflow',                   'cat': 'SSMP'},
    26: {'point': ['P101','LIT301'],          'stage': [1,3],  'type': 'mixed',    'actual_change': True,  'impact': True,  'desc': 'P-101 on continuously; Spoof LIT-301',              'intent': 'T101 underflow; T301 overflow',    'cat': 'MSSP'},
    27: {'point': ['P302','LIT401'],          'stage': [3,4],  'type': 'mixed',    'actual_change': True,  'impact': True,  'desc': 'Keep P-302 on; Spoof LIT-401',                      'intent': 'Tank overflow',                    'cat': 'MSSP'},
    28: {'point': ['P302'],                   'stage': [3],    'type': 'actuator', 'actual_change': True,  'impact': True,  'desc': 'Close P-302',                                       'intent': 'Stop inflow of T-401',             'cat': 'SSMP'},
    29: {'point': ['P201', 'P203', 'P205'],   'stage': [2],    'type': 'actuator', 'actual_change': True,  'impact': False, 'desc': 'Wastage of chemicals',                              'intent': 'The three dosing pump did not start because of some mechanical interloc', 'cat': 'SSMP'},
    30: {'point': ['LIT101','P101','MV201'],  'stage': [1,2],  'type': 'mixed',    'actual_change': True,  'impact': True,  'desc': 'P-101 on; Open MV-101; Spoof LIT-101, MV-201',      'intent': 'T101 underflow; T301 overflow',    'cat': 'MSMP'},
    31: {'point': ['LIT401'],                 'stage': [4],    'type': 'sensor',   'actual_change': False, 'impact': True,  'desc': 'Set LIT-401 < L',                                   'intent': 'Tank overflow',                    'cat': 'SSSP'},
    32: {'point': ['LIT301'],                 'stage': [3],    'type': 'sensor',   'actual_change': False, 'impact': True,  'desc': 'Set LIT-301 > HH',                                  'intent': 'Tank underflow; Damage P-302',     'cat': 'SSSP'},
    33: {'point': ['LIT101'],                 'stage': [1],    'type': 'sensor',   'actual_change': False, 'impact': True,  'desc': 'Set LIT-101 > H',                                   'intent': 'Tank underflow; Damage P-101',     'cat': 'SSSP'},
    34: {'point': ['P101'],                   'stage': [1],    'type': 'actuator', 'actual_change': True,  'impact': True,  'desc': 'Turn P-101 off',                                    'intent': 'Stops outflow',                    'cat': 'SSSP'},
    35: {'point': ['P101','P102'],            'stage': [1],    'type': 'actuator', 'actual_change': True,  'impact': True,  'desc': 'Turn P-101 off; Keep P-102 off',                    'intent': 'Stops outflow',                    'cat': 'SSMP'},
    36: {'point': ['LIT101'],                 'stage': [1],    'type': 'sensor',   'actual_change': False, 'impact': True,  'desc': 'Set LIT-101 < LL',                                  'intent': 'Tank overflow',                    'cat': 'SSSP'},
    37: {'point': ['P501','FIT502'],          'stage': [5],    'type': 'mixed',    'actual_change': True,  'impact': True,  'desc': 'Close P-501; Set FIT-502=1.29',                     'intent': 'Reduced output',                   'cat': 'SSMP'},
    38: {'point': ['AIT402','AIT502'],        'stage': [4,5],  'type': 'sensor',   'actual_change': False, 'impact': True,  'desc': 'Set AIT-402=260; Set AIT-502=260',                  'intent': 'Water to drain (overdosing)',      'cat': 'MSSP'},
    39: {'point': ['FIT401','AIT502'],        'stage': [4,5],  'type': 'sensor',   'actual_change': False, 'impact': True,  'desc': 'Set FIT-401=0.5; Set AIT-502<10',                   'intent': 'UV shutdown; water to RO',         'cat': 'MSSP'},
    40: {'point': ['FIT401'],                 'stage': [4],    'type': 'sensor',   'actual_change': False, 'impact': True,  'desc': 'Set FIT-401 = 0',                                   'intent': 'UV shutdown; water to RO',         'cat': 'SSSP'},
    41: {'point': ['LIT301'],                 'stage': [3],    'type': 'sensor',   'actual_change': False, 'impact': True,  'desc': 'Decrease LIT-301 -0.5mm/s',                         'intent': 'Tank overflow',                    'cat': 'SSSP'},
}

Sensors = [
    'FIT101', 'LIT101',
    'AIT201', 'AIT202', 'AIT203', 'FIT201',
    'DPIT301', 'FIT301', 'LIT301',
    'AIT401', 'AIT402', 'FIT401', 'LIT401',
    'AIT501', 'AIT502', 'AIT503', 'AIT504',
    'FIT501', 'FIT502', 'FIT503', 'FIT504',
    'PIT501', 'PIT502', 'PIT503',
    'FIT601',
]

Actuators = [
    'MV101', 'P101', 'P102',
    'MV201', 'P201', 'P202', 'P203', 'P204', 'P205', 'P206',
    'MV301', 'MV302', 'MV303', 'MV304', 'P301', 'P302',
    'P401', 'P402', 'P403', 'P404', 'UV401',
    'P501', 'P502',
    'P601', 'P602', 'P603',
]

stage_map = {
    1: {'sensors': ['FIT101', 'LIT101'],
        'actuators': ['MV101', 'P101', 'P102']},
    2: {'sensors': ['AIT201', 'AIT202', 'AIT203', 'FIT201'],
        'actuators': ['MV201', 'P201', 'P202', 'P203', 'P204', 'P205', 'P206']},
    3: {'sensors': ['DPIT301', 'FIT301', 'LIT301'],
        'actuators': ['MV301', 'MV302', 'MV303', 'MV304', 'P301', 'P302']},
    4: {'sensors': ['AIT401', 'AIT402', 'FIT401', 'LIT401'],
        'actuators': ['P401', 'P402', 'P403', 'P404', 'UV401']},
    5: {'sensors': ['AIT501', 'AIT502', 'AIT503', 'AIT504',
                    'FIT501', 'FIT502', 'FIT503', 'FIT504',
                    'PIT501', 'PIT502', 'PIT503'],
        'actuators': ['P501', 'P502']},
    6: {'sensors': ['FIT601'],
        'actuators': ['P601', 'P602', 'P603']},
}

attacks_time = [
    ("Attack1",  "2015-12-28 10:29:14", "2015-12-28 10:44:53"),
    ("Attack2",  "2015-12-28 10:51:08", "2015-12-28 10:58:30"),
    ("Attack3",  "2015-12-28 11:22:00", "2015-12-28 11:28:22"),
    ("Attack4",  "2015-12-28 11:47:39", "2015-12-28 11:54:08"),
    ("Attack5",  "2015-12-28 11:58:20", "2015-12-28 12:00:54"),  # No Impact
    ("Attack6",  "2015-12-28 12:00:55", "2015-12-28 12:04:10"),
    ("Attack7",  "2015-12-28 12:08:25", "2015-12-28 12:15:33"),
    ("Attack8",  "2015-12-28 13:10:10", "2015-12-28 13:26:13"),
    ("Attack9",  "2015-12-28 14:15:00", "2015-12-28 14:16:19"),  # No Impact
    ("Attack10", "2015-12-28 14:16:20", "2015-12-28 14:18:59"),
    ("Attack11", "2015-12-28 14:19:00", "2015-12-28 14:28:20"),
    ("Attack12", "2015-12-29 11:10:40", "2015-12-29 11:11:24"),  # No Impact
    ("Attack13", "2015-12-29 11:11:25", "2015-12-29 11:15:17"),
    ("Attack14", "2015-12-29 11:35:40", "2015-12-29 11:42:50"),
    ("Attack15", "2015-12-29 11:52:01", "2015-12-29 11:57:24"),  # No Impact
    ("Attack16", "2015-12-29 11:57:25", "2015-12-29 12:02:00"),
    ("Attack17", "2015-12-29 14:38:12", "2015-12-29 14:50:08"),
    ("Attack18", "2015-12-29 18:08:55", "2015-12-29 18:10:42"),  # No Impact
    ("Attack19", "2015-12-29 18:10:43", "2015-12-29 18:15:01"),
    ("Attack20", "2015-12-29 18:15:43", "2015-12-29 18:22:17"),
    ("Attack21", "2015-12-29 18:30:00", "2015-12-29 18:42:00"),
    ("Attack22", "2015-12-29 22:55:18", "2015-12-29 23:03:00"),
    ("Attack23", "2015-12-30 01:42:34", "2015-12-30 01:54:10"),
    ("Attack24", "2015-12-30 09:51:08", "2015-12-30 09:56:28"),
    ("Attack25", "2015-12-30 10:01:50", "2015-12-30 10:12:01"),
    ("Attack26", "2015-12-30 17:04:56", "2015-12-30 17:29:00"),
    ("Attack27", "2015-12-31 01:17:08", "2015-12-31 01:45:18"),
    ("Attack28", "2015-12-31 01:45:19", "2015-12-31 11:15:27"),
    ("Attack29", "2015-12-31 15:32:00", "2015-12-31 15:34:00"),
    ("Attack30", "2015-12-31 15:47:40", "2015-12-31 16:07:10"),
    ("Attack31", "2015-12-31 22:05:34", "2015-12-31 22:11:40"),
    ("Attack32", "2016-01-01 10:36:00", "2016-01-01 10:46:00"),
    ("Attack33", "2016-01-01 14:21:12", "2016-01-01 14:28:35"),
    ("Attack34", "2016-01-01 17:12:40", "2016-01-01 17:14:20"),
    ("Attack35", "2016-01-01 17:18:56", "2016-01-01 17:26:56"),
    ("Attack36", "2016-01-01 22:16:01", "2016-01-01 22:25:00"),
    ("Attack37", "2016-01-02 11:17:02", "2016-01-02 11:24:50"),
    ("Attack38", "2016-01-02 11:31:38", "2016-01-02 11:36:18"),
    ("Attack39", "2016-01-02 11:43:48", "2016-01-02 11:50:28"),
    ("Attack40", "2016-01-02 11:51:42", "2016-01-02 11:56:38"),
    ("Attack41", "2016-01-02 13:13:02", "2016-01-02 13:40:56")
]

