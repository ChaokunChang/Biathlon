ALL_REG_TASKS = [
    "trips",
    "tripsfeast",
    "tick",
    "tickv2",
    "battery", "batterytest",
    "batteryv2",
    "turbofan",
    "turbofanall",
    "tripsralf", "tripsralfv2",
    "tripsralftest", "tripsralf2h",
    "tickralftest", "tickralf",
    "tickralfv2test", "tickralfv2"
]
ALL_CLS_TASKS = [
    "tdfraudralftest", "tdfraudralf",
    "tdfraudralf2h", "tdfraudralf2d",
    "tdfraudralftestv2", "tdfraudralfv2",
    "tdfraudralf2hv2", "tdfraudralf2dv2",
    "machineryralftest", "machineryralf",
    "cheaptrips",
    "cheaptripsfeast",
    "machinery",
    "ccfraud",
    "machinerymulti",
    "tdfraud",
    "tdfraudrandom",
    "tdfraudkaggle",
    "student", "studentqnotest",
    "studentqnov2", "studentqnov2subset", "studentqnov2test",
    "performance12", "performance16", "performance17",
]

StudentQNo = [f"studentqno{i}" for i in range(1, 19)]
ALL_CLS_TASKS += StudentQNo

StudentQNo18VaryNF = [f"studentqno18nf{i}" for i in range(1, 13)]
ALL_CLS_TASKS += StudentQNo18VaryNF

MachineryVaryNF = [f"machinerynf{i}" for i in range(1, 8)] + [
    f"machineryxf{i}" for i in range(1, 9)
]
MachineryMultiVaryNF = [f"machinerymultif{i}" for i in range(1, 8)] + [
    f"machinerymultixf{i}" for i in range(1, 9)
]
ALL_CLS_TASKS += MachineryVaryNF
ALL_CLS_TASKS += MachineryMultiVaryNF

MachineryRalfVaryNF = [f"machineryralfnf{i}" for i in range(1, 8)]
ALL_CLS_TASKS += MachineryRalfVaryNF

TickVaryNMonths = [f"tickvaryNM{i}" for i in range(1, 30)]
ALL_REG_TASKS += TickVaryNMonths

TripsFeastVaryWindow = [f"tripsfeastw{i}" for i in range(1, 30)]
ALL_CLS_TASKS += TripsFeastVaryWindow
