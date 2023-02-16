const videodata = {
    groups: [
        {
            title: "Method: FMLP",
            columns: [
                { key: "video_id", name: "ID" },
                { key: "parameters.sxy", name: "Spatial coordinate scale s_x"},
                { key: "parameters.st", name: "Temporal coordinate scale s_t" },
                { key: "parameters.hidden_layers", name: "Hidden layers" },
                { key: "parameters.width", name: "Layer width" },
                { key: "parameters.num_lines", name: "Number of lines" },
                { key: "performance.max_ser", name: "SER" },
                { key: "performance.max_ser_epoch", name: "epochs" },
                { key: "performance.max_ser_time", name: "training time" },
            ],
            videos: [
                {
                    "parameters": {
                        "num_lines": 6,
                        "K": 225,
                        "sxy": 33.3,
                        "st": 5.3,
                        "hidden_layers": 3,
                        "width": 512
                    },
                    "method": "FMLP",
                    "video_id": 1,
                    "title": "FMLP 1",
                    "src": "static/cinemri/export/cava_v1/10/MultiResFMLP/validation/225/paper_implementation, s_t 5.3 spatial_coordinate_scales 200 mlp_hidden_layers 3/timecoded_vfr.mp4",
                    "selected": false,
                    "performance": {
                        "max_ser": 16.96,
                        "max_ser_epoch": 733,
                        "max_ser_time": 144
                    }
                },
                {
                    "parameters": {
                        "num_lines": 6,
                        "K": 225,
                        "sxy": 75.0,
                        "st": 5.3,
                        "hidden_layers": 9,
                        "width": 512
                    },
                    "method": "FMLP",
                    "video_id": 2,
                    "title": "FMLP 2",
                    "src": "static/cinemri/export/cava_v1/10/MultiResFMLP/validation/225/paper_implementation, s_t 5.3 spatial_coordinate_scales 450 mlp_hidden_layers 9 (2)/timecoded_vfr.mp4",
                    "selected": false,
                    "performance": {
                        "max_ser": 16.86,
                        "max_ser_epoch": 206,
                        "max_ser_time": 101
                    }
                },
                {
                    "parameters": {
                        "num_lines": 6,
                        "K": 225,
                        "sxy": 75.0,
                        "st": 5.3,
                        "hidden_layers": 9,
                        "width": 512
                    },
                    "method": "FMLP",
                    "video_id": 3,
                    "title": "FMLP 3",
                    "src": "static/cinemri/export/cava_v1/10/MultiResFMLP/validation/225/paper_implementation, s_t 5.3 spatial_coordinate_scales 450 mlp_hidden_layers 9/timecoded_vfr.mp4",
                    "selected": false,
                    "performance": {
                        "max_ser": 17.02,
                        "max_ser_epoch": 363,
                        "max_ser_time": 178
                    }
                },
                {
                    "parameters": {
                        "num_lines": 6,
                        "K": 225,
                        "sxy": 33.3,
                        "st": 5.3,
                        "hidden_layers": 9,
                        "width": 512
                    },
                    "method": "FMLP",
                    "video_id": 4,
                    "title": "FMLP 4",
                    "src": "static/cinemri/export/cava_v1/10/MultiResFMLP/validation/225/paper_implementation, s_t 5.3 spatial_coordinate_scales 200 mlp_hidden_layers 9/timecoded_vfr.mp4",
                    "selected": false,
                    "performance": {
                        "max_ser": 17.08,
                        "max_ser_epoch": 570,
                        "max_ser_time": 284
                    }
                },
                {
                    "parameters": {
                        "num_lines": 6,
                        "K": 225,
                        "sxy": 33.3,
                        "st": 5.3,
                        "hidden_layers": 5,
                        "width": 256
                    },
                    "method": "FMLP",
                    "video_id": 5,
                    "title": "FMLP 5",
                    "src": "static/cinemri/export/cava_v1/10/MultiResFMLP/validation/225/paper_implementation, s_t 5.3 spatial_coordinate_scales 200 mlp_hidden_layers 5 width 256/timecoded_vfr.mp4",
                    "selected": false,
                    "performance": {
                        "max_ser": 17.08,
                        "max_ser_epoch": 944,
                        "max_ser_time": 193
                    }
                },
                {
                    "parameters": {
                        "num_lines": 6,
                        "K": 225,
                        "sxy": 33.3,
                        "st": 5.3,
                        "hidden_layers": 9,
                        "width": 512
                    },
                    "method": "FMLP",
                    "video_id": 6,
                    "title": "FMLP 6",
                    "src": "static/cinemri/export/cava_v1/10/MultiResFMLP/validation/225/paper_implementation, s_t 5.3 spatial_coordinate_scales 200 mlp_hidden_layers 9 (2)/timecoded_vfr.mp4",
                    "selected": false,
                    "performance": {
                        "max_ser": 17.08,
                        "max_ser_epoch": 570,
                        "max_ser_time": 283
                    }
                },
                {
                    "parameters": {
                        "num_lines": 6,
                        "K": 225,
                        "sxy": 33.3,
                        "st": 5.3,
                        "hidden_layers": 5,
                        "width": 128
                    },
                    "method": "FMLP",
                    "video_id": 7,
                    "title": "FMLP 7",
                    "src": "static/cinemri/export/cava_v1/10/MultiResFMLP/validation/225/paper_implementation, s_t 5.3 spatial_coordinate_scales 200 mlp_hidden_layers 5 width 128/timecoded_vfr.mp4",
                    "selected": false,
                    "performance": {
                        "max_ser": 16.99,
                        "max_ser_epoch": 1934,
                        "max_ser_time": 261
                    }
                },
                {
                    "parameters": {
                        "num_lines": 6,
                        "K": 225,
                        "sxy": 33.3,
                        "st": 5.3,
                        "hidden_layers": 5,
                        "width": 128
                    },
                    "method": "FMLP",
                    "video_id": 8,
                    "title": "FMLP 8",
                    "src": "static/cinemri/export/cava_v1/10/MultiResFMLP/validation/225/paper_implementation, s_t 5.3 spatial_coordinate_scales 200 mlp_hidden_layers 5 width 128 (2)/timecoded_vfr.mp4",
                    "selected": false,
                    "performance": {
                        "max_ser": 16.9,
                        "max_ser_epoch": 967,
                        "max_ser_time": 101
                    }
                },
                {
                    "parameters": {
                        "num_lines": 6,
                        "K": 225,
                        "sxy": 33.3,
                        "st": 5.3,
                        "hidden_layers": 5,
                        "width": 768
                    },
                    "method": "FMLP",
                    "video_id": 9,
                    "title": "FMLP 9",
                    "src": "static/cinemri/export/cava_v1/10/MultiResFMLP/validation/225/paper_implementation, s_t 5.3 spatial_coordinate_scales 200 mlp_hidden_layers 5 width 768/timecoded_vfr.mp4",
                    "selected": false,
                    "performance": {
                        "max_ser": 17.0,
                        "max_ser_epoch": 559,
                        "max_ser_time": 276
                    }
                },
                {
                    "parameters": {
                        "num_lines": 6,
                        "K": 225,
                        "sxy": 33.3,
                        "st": 5.3,
                        "hidden_layers": 7,
                        "width": 512
                    },
                    "method": "FMLP",
                    "video_id": 10,
                    "title": "FMLP 10",
                    "src": "static/cinemri/export/cava_v1/10/MultiResFMLP/validation/225/paper_implementation, s_t 5.3 spatial_coordinate_scales 200 combined ff/timecoded_vfr.mp4",
                    "selected": false,
                    "performance": {
                        "max_ser": 17.05,
                        "max_ser_epoch": 493,
                        "max_ser_time": 195
                    }
                },
                {
                    "parameters": {
                        "num_lines": 6,
                        "K": 225,
                        "sxy": 33.3,
                        "st": 5.3,
                        "hidden_layers": 7,
                        "width": 512
                    },
                    "method": "FMLP",
                    "video_id": 11,
                    "title": "FMLP 11",
                    "src": "static/cinemri/export/cava_v1/10/MultiResFMLP/validation/225/paper_implementation, s_t 5.3/timecoded_vfr.mp4",
                    "selected": false,
                    "performance": {
                        "max_ser": 17.05,
                        "max_ser_epoch": 493,
                        "max_ser_time": 195
                    }
                },
                {
                    "parameters": {
                        "num_lines": 6,
                        "K": 225,
                        "sxy": 33.3,
                        "st": 5.3,
                        "hidden_layers": 5,
                        "width": 512
                    },
                    "method": "FMLP",
                    "video_id": 12,
                    "title": "FMLP 12",
                    "src": "static/cinemri/export/cava_v1/10/MultiResFMLP/validation/225/paper_implementation, s_t 5.3 spatial_coordinate_scales 200 mlp_hidden_layers 5/timecoded_vfr.mp4",
                    "selected": false,
                    "performance": {
                        "max_ser": 17.04,
                        "max_ser_epoch": 449,
                        "max_ser_time": 130
                    }
                },
                {
                    "parameters": {
                        "num_lines": 6,
                        "K": 225,
                        "sxy": 33.3,
                        "st": 5.3,
                        "hidden_layers": 5,
                        "width": 384
                    },
                    "method": "FMLP",
                    "video_id": 13,
                    "title": "FMLP 13",
                    "src": "static/cinemri/export/cava_v1/10/MultiResFMLP/validation/225/paper_implementation, s_t 5.3 spatial_coordinate_scales 200 mlp_hidden_layers 5 width 384/timecoded_vfr.mp4",
                    "selected": false,
                    "performance": {
                        "max_ser": 17.05,
                        "max_ser_epoch": 577,
                        "max_ser_time": 124
                    }
                },
                {
                    "parameters": {
                        "num_lines": 6,
                        "K": 225,
                        "sxy": 75.0,
                        "st": 5.3,
                        "hidden_layers": 5,
                        "width": 512
                    },
                    "method": "FMLP",
                    "video_id": 14,
                    "title": "FMLP 14",
                    "src": "static/cinemri/export/cava_v1/10/MultiResFMLP/validation/225/paper_implementation, s_t 5.3 spatial_coordinate_scales 450 mlp_hidden_layers 5/timecoded_vfr.mp4",
                    "selected": false,
                    "performance": {
                        "max_ser": 17.02,
                        "max_ser_epoch": 319,
                        "max_ser_time": 92
                    }
                },
                {
                    "parameters": {
                        "num_lines": 6,
                        "K": 225,
                        "sxy": 33.3,
                        "st": 5.3,
                        "hidden_layers": 5,
                        "width": 512
                    },
                    "method": "FMLP",
                    "video_id": 15,
                    "title": "FMLP 15",
                    "src": "static/cinemri/export/cava_v1/10/MultiResFMLP/validation/225/paper_implementation, s_t 5.3 spatial_coordinate_scales 200 mlp_hidden_layers 5 width 512/timecoded_vfr.mp4",
                    "selected": false,
                    "performance": {
                        "max_ser": 17.04,
                        "max_ser_epoch": 449,
                        "max_ser_time": 130
                    }
                },
                {
                    "parameters": {
                        "num_lines": 6,
                        "K": 225,
                        "sxy": 75.0,
                        "st": 5.3,
                        "hidden_layers": 3,
                        "width": 512
                    },
                    "method": "FMLP",
                    "video_id": 16,
                    "title": "FMLP 16",
                    "src": "static/cinemri/export/cava_v1/10/MultiResFMLP/validation/225/paper_implementation, s_t 5.3 spatial_coordinate_scales 450 mlp_hidden_layers 3/timecoded_vfr.mp4",
                    "selected": false,
                    "performance": {
                        "max_ser": 16.96,
                        "max_ser_epoch": 548,
                        "max_ser_time": 105
                    }
                },
                {
                    "parameters": {
                        "num_lines": 6,
                        "K": 225,
                        "sxy": 50.0,
                        "st": 5.3,
                        "hidden_layers": 7,
                        "width": 512
                    },
                    "method": "FMLP",
                    "video_id": 17,
                    "title": "FMLP 17",
                    "src": "static/cinemri/export/cava_v1/10/MultiResFMLP/validation/225/spatial_coordinate_scales/paper_implementation, s_t 5.3 spatial_coordinate_scales 300/timecoded_vfr.mp4",
                    "selected": false,
                    "performance": {
                        "max_ser": 17.05,
                        "max_ser_epoch": 400,
                        "max_ser_time": 156
                    }
                },
                {
                    "parameters": {
                        "num_lines": 6,
                        "K": 225,
                        "sxy": 40.0,
                        "st": 5.3,
                        "hidden_layers": 7,
                        "width": 512
                    },
                    "method": "FMLP",
                    "video_id": 18,
                    "title": "FMLP 18",
                    "src": "static/cinemri/export/cava_v1/10/MultiResFMLP/validation/225/spatial_coordinate_scales/paper_implementation, s_t 5.3 spatial_coordinate_scales 240/timecoded_vfr.mp4",
                    "selected": false,
                    "performance": {
                        "max_ser": 17.05,
                        "max_ser_epoch": 412,
                        "max_ser_time": 163
                    }
                },
                {
                    "parameters": {
                        "num_lines": 6,
                        "K": 225,
                        "sxy": 5.0,
                        "st": 5.3,
                        "hidden_layers": 7,
                        "width": 512
                    },
                    "method": "FMLP",
                    "video_id": 19,
                    "title": "FMLP 19",
                    "src": "static/cinemri/export/cava_v1/10/MultiResFMLP/validation/225/spatial_coordinate_scales/paper_implementation, s_t 5.3 spatial_coordinate_scales 30.0/timecoded_vfr.mp4",
                    "selected": false,
                    "performance": {
                        "max_ser": 17.09,
                        "max_ser_epoch": 1743,
                        "max_ser_time": 734
                    }
                },
                {
                    "parameters": {
                        "num_lines": 6,
                        "K": 225,
                        "sxy": 2.5,
                        "st": 5.3,
                        "hidden_layers": 7,
                        "width": 512
                    },
                    "method": "FMLP",
                    "video_id": 20,
                    "title": "FMLP 20",
                    "src": "static/cinemri/export/cava_v1/10/MultiResFMLP/validation/225/spatial_coordinate_scales/paper_implementation, s_t 5.3 spatial_coordinate_scales 15.0/timecoded_vfr.mp4",
                    "selected": false,
                    "performance": {
                        "max_ser": 17.11,
                        "max_ser_epoch": 2408,
                        "max_ser_time": 1060
                    }
                },
                {
                    "parameters": {
                        "num_lines": 6,
                        "K": 225,
                        "sxy": 15.0,
                        "st": 5.3,
                        "hidden_layers": 7,
                        "width": 512
                    },
                    "method": "FMLP",
                    "video_id": 21,
                    "title": "FMLP 21",
                    "src": "static/cinemri/export/cava_v1/10/MultiResFMLP/validation/225/spatial_coordinate_scales/paper_implementation, s_t 5.3 spatial_coordinate_scales 90.0/timecoded_vfr.mp4",
                    "selected": false,
                    "performance": {
                        "max_ser": 17.09,
                        "max_ser_epoch": 745,
                        "max_ser_time": 530
                    }
                },
                {
                    "parameters": {
                        "num_lines": 6,
                        "K": 225,
                        "sxy": 75.0,
                        "st": 5.3,
                        "hidden_layers": 7,
                        "width": 512
                    },
                    "method": "FMLP",
                    "video_id": 22,
                    "title": "FMLP 22",
                    "src": "static/cinemri/export/cava_v1/10/MultiResFMLP/validation/225/spatial_coordinate_scales/paper_implementation, s_t 5.3 spatial_coordinate_scales 450/timecoded_vfr.mp4",
                    "selected": false,
                    "performance": {
                        "max_ser": 17.04,
                        "max_ser_epoch": 294,
                        "max_ser_time": 115
                    }
                },
                {
                    "parameters": {
                        "num_lines": 6,
                        "K": 225,
                        "sxy": 100.0,
                        "st": 5.3,
                        "hidden_layers": 7,
                        "width": 512
                    },
                    "method": "FMLP",
                    "video_id": 23,
                    "title": "FMLP 23",
                    "src": "static/cinemri/export/cava_v1/10/MultiResFMLP/validation/225/spatial_coordinate_scales/paper_implementation, s_t 5.3 spatial_coordinate_scales 600/timecoded_vfr.mp4",
                    "selected": false,
                    "performance": {
                        "max_ser": 17.02,
                        "max_ser_epoch": 467,
                        "max_ser_time": 185
                    }
                },
                {
                    "parameters": {
                        "num_lines": 6,
                        "K": 225,
                        "sxy": 150.0,
                        "st": 5.3,
                        "hidden_layers": 7,
                        "width": 512
                    },
                    "method": "FMLP",
                    "video_id": 24,
                    "title": "FMLP 24",
                    "src": "static/cinemri/export/cava_v1/10/MultiResFMLP/validation/225/spatial_coordinate_scales/paper_implementation, s_t 5.3 spatial_coordinate_scales 900/timecoded_vfr.mp4",
                    "selected": false,
                    "performance": {
                        "max_ser": 17.02,
                        "max_ser_epoch": 314,
                        "max_ser_time": 123
                    }
                },
                {
                    "parameters": {
                        "num_lines": 6,
                        "K": 225,
                        "sxy": 1.5,
                        "st": 5.3,
                        "hidden_layers": 7,
                        "width": 512
                    },
                    "method": "FMLP",
                    "video_id": 25,
                    "title": "FMLP 25",
                    "src": "static/cinemri/export/cava_v1/10/MultiResFMLP/validation/225/spatial_coordinate_scales/paper_implementation, s_t 5.3 spatial_coordinate_scales 9.0/timecoded_vfr.mp4",
                    "selected": false,
                    "performance": {
                        "max_ser": 17.1,
                        "max_ser_epoch": 2908,
                        "max_ser_time": 1298
                    }
                },
                {
                    "parameters": {
                        "num_lines": 6,
                        "K": 225,
                        "sxy": 33.3,
                        "st": 20.0,
                        "hidden_layers": 7,
                        "width": 512
                    },
                    "method": "FMLP",
                    "video_id": 26,
                    "title": "FMLP 26",
                    "src": "static/cinemri/export/cava_v1/10/MultiResFMLP/validation/225/temporal_coordinate_scale/paper_implementation, s_t 20.0 spatial_coordinate_scales 200/timecoded_vfr.mp4",
                    "selected": false,
                    "performance": {
                        "max_ser": 16.94,
                        "max_ser_epoch": 331,
                        "max_ser_time": 129
                    }
                },
                {
                    "parameters": {
                        "num_lines": 6,
                        "K": 225,
                        "sxy": 33.3,
                        "st": 10.0,
                        "hidden_layers": 7,
                        "width": 512
                    },
                    "method": "FMLP",
                    "video_id": 27,
                    "title": "FMLP 27",
                    "src": "static/cinemri/export/cava_v1/10/MultiResFMLP/validation/225/temporal_coordinate_scale/paper_implementation, s_t 10.0 spatial_coordinate_scales 200/timecoded_vfr.mp4",
                    "selected": false,
                    "performance": {
                        "max_ser": 17.0,
                        "max_ser_epoch": 487,
                        "max_ser_time": 192
                    }
                },
                {
                    "parameters": {
                        "num_lines": 6,
                        "K": 225,
                        "sxy": 33.3,
                        "st": 7.0,
                        "hidden_layers": 7,
                        "width": 512
                    },
                    "method": "FMLP",
                    "video_id": 28,
                    "title": "FMLP 28",
                    "src": "static/cinemri/export/cava_v1/10/MultiResFMLP/validation/225/temporal_coordinate_scale/paper_implementation, s_t 7.0 spatial_coordinate_scales 200/timecoded_vfr.mp4",
                    "selected": false,
                    "performance": {
                        "max_ser": 17.04,
                        "max_ser_epoch": 569,
                        "max_ser_time": 229
                    }
                },
                {
                    "parameters": {
                        "num_lines": 6,
                        "K": 225,
                        "sxy": 33.3,
                        "st": 2.5,
                        "hidden_layers": 7,
                        "width": 512
                    },
                    "method": "FMLP",
                    "video_id": 29,
                    "title": "FMLP 29",
                    "src": "static/cinemri/export/cava_v1/10/MultiResFMLP/validation/225/temporal_coordinate_scale/paper_implementation, s_t 2.5 spatial_coordinate_scales 200.0/timecoded_vfr.mp4",
                    "selected": false,
                    "performance": {
                        "max_ser": 17.14,
                        "max_ser_epoch": 743,
                        "max_ser_time": 544
                    }
                },
                {
                    "parameters": {
                        "num_lines": 6,
                        "K": 225,
                        "sxy": 33.3,
                        "st": 1.0,
                        "hidden_layers": 7,
                        "width": 512
                    },
                    "method": "FMLP",
                    "video_id": 30,
                    "title": "FMLP 30",
                    "src": "static/cinemri/export/cava_v1/10/MultiResFMLP/validation/225/temporal_coordinate_scale/paper_implementation, s_t 1.0 spatial_coordinate_scales 200.0/timecoded_vfr.mp4",
                    "selected": true,
                    "performance": {
                        "max_ser": 17.16,
                        "max_ser_epoch": 804,
                        "max_ser_time": 324
                    }
                },
                {
                    'parameters': {
                        'num_lines': 6,
                        'K': 225,
                        'sxy': 33.3,
                        'st': 0.3,
                        'hidden_layers': 7,
                        'width': 512
                    },
                    'method': 'FMLP',
                    'video_id': 31,
                    'title': 'FMLP 31',
                    'src': 'static/cinemri/export/cava_v1/10/MultiResFMLP/validation/225/temporal_coordinate_scale/paper_implementation, s_t 0.3 spatial_coordinate_scales 200.0/timecoded_vfr.mp4',
                    'selected': false,
                    'performance': {
                        'max_ser': 17.12,
                        'max_ser_epoch': 1297,
                        'max_ser_time': 532
                    }
                },
                {
                    "parameters": {
                        "num_lines": 6,
                        "K": 225,
                        "sxy": 33.3,
                        "st": 0.1,
                        "hidden_layers": 7,
                        "width": 512
                    },
                    "method": "FMLP",
                    "video_id": 32,
                    "title": "FMLP 32",
                    "src": "static/cinemri/export/cava_v1/10/MultiResFMLP/validation/225/temporal_coordinate_scale/paper_implementation, s_t 0.1 spatial_coordinate_scales 200.0/timecoded_vfr.mp4",
                    "selected": false,
                    "performance": {
                        "max_ser": 16.99,
                        "max_ser_epoch": 1799,
                        "max_ser_time": 757
                    }
                }
            ]
        },
        {
            title: "Method: t-DIP",
            columns: [
                { key: "video_id", name: "ID" },
                { key: "parameters.conv_channels", name: "Channel depth" },
                { key: "parameters.z", name: "z_slack" },
                { key: "parameters.num_lines", name: "Number of lines" },
                { key: "performance.max_ser", name: "SER" },
                { key: "performance.max_ser_epoch", name: "epochs" },
                { key: "performance.max_ser_time", name: "training time" },
            ],
            videos: [
                {
                    "parameters": {
                        "num_lines": 6,
                        "K": 225,
                        "conv_channels": 128,
                        "z": 1.0
                    },
                    "method": "t-DIP",
                    "video_id": 1,
                    "title": "t-DIP 1",
                    "src": "static/cinemri/export/cava_v1/10/TimedependentDIP/validation/225/z_slack 1*1.0, trajectory frame time/timecoded_vfr.mp4",
                    "selected": false,
                    "performance": {
                        "max_ser": 17.1,
                        "max_ser_epoch": 241,
                        "max_ser_time": 21
                    }
                },
                {
                    "parameters": {
                        "num_lines": 6,
                        "K": 225,
                        "conv_channels": 512,
                        "z": 1.0
                    },
                    "method": "t-DIP",
                    "video_id": 2,
                    "title": "t-DIP 2",
                    "src": "static/cinemri/export/cava_v1/10/TimedependentDIP/validation/225/z_slack 1.0, trajectory frame time, conv_channels 512/timecoded_vfr.mp4",
                    "selected": false,
                    "performance": {
                        "max_ser": 17.11,
                        "max_ser_epoch": 183,
                        "max_ser_time": 90
                    }
                },
                {
                    "parameters": {
                        "num_lines": 6,
                        "K": 225,
                        "conv_channels": 256,
                        "z": 1.0
                    },
                    "method": "t-DIP",
                    "video_id": 3,
                    "title": "t-DIP 3",
                    "src": "static/cinemri/export/cava_v1/10/TimedependentDIP/validation/225/z_slack 1.0, trajectory frame time, conv_channels 256/timecoded_vfr.mp4",
                    "selected": false,
                    "performance": {
                        "max_ser": 17.09,
                        "max_ser_epoch": 159,
                        "max_ser_time": 26
                    }
                },
                {
                    "parameters": {
                        "num_lines": 6,
                        "K": 225,
                        "conv_channels": 128,
                        "z": 0.1
                    },
                    "method": "t-DIP",
                    "video_id": 4,
                    "title": "t-DIP 4",
                    "src": "static/cinemri/export/cava_v1/10/TimedependentDIP/validation/225/z_slack 0.1, trajectory frame time, conv_channels 128/timecoded_vfr.mp4",
                    "selected": false,
                    "performance": {
                        "max_ser": 17.15,
                        "max_ser_epoch": 329,
                        "max_ser_time": 26
                    }
                },
                {
                    "parameters": {
                        "num_lines": 6,
                        "K": 225,
                        "conv_channels": 256,
                        "z": 0.1
                    },
                    "method": "t-DIP",
                    "video_id": 5,
                    "title": "t-DIP 5",
                    "src": "static/cinemri/export/cava_v1/10/TimedependentDIP/validation/225/z_slack 0.1, trajectory frame time, conv_channels 256/timecoded_vfr.mp4",
                    "selected": true,
                    "performance": {
                        "max_ser": 17.19,
                        "max_ser_epoch": 252,
                        "max_ser_time": 42
                    }
                },
                {
                    "parameters": {
                        "num_lines": 6,
                        "K": 225,
                        "conv_channels": 64,
                        "z": 1.0
                    },
                    "method": "t-DIP",
                    "video_id": 6,
                    "title": "t-DIP 6",
                    "src": "static/cinemri/export/cava_v1/10/TimedependentDIP/validation/225/z_slack 1.0, trajectory frame time, conv_channels 64/timecoded_vfr.mp4",
                    "selected": false,
                    "performance": {
                        "max_ser": 17.1,
                        "max_ser_epoch": 824,
                        "max_ser_time": 60
                    }
                },
                {
                    "parameters": {
                        "num_lines": 6,
                        "K": 225,
                        "conv_channels": 128,
                        "z": 0.5
                    },
                    "method": "t-DIP",
                    "video_id": 7,
                    "title": "t-DIP 7",
                    "src": "static/cinemri/export/cava_v1/10/TimedependentDIP/validation/225/z_slack 0.5, trajectory frame time, conv_channels 128/timecoded_vfr.mp4",
                    "selected": false,
                    "performance": {
                        "max_ser": 17.09,
                        "max_ser_epoch": 250,
                        "max_ser_time": 19
                    }
                },
                {
                    "src": "static/cinemri/export/cava_v1/10/BHReference/validation/225/timecoded_vfr.mp4",
                    "title": "BH reference",
                    "selected": true,
                    "listed": false
                }
            ]
        }
    ]
}

module.exports = videodata

