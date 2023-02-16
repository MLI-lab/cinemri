const chartdata = {
    type: 'line',
    data: {
        labels: [1.5, 2.5, 5.0, 15.0, 33.3, 40.0, 50.0, 75.0, 100.0, 150.0],
        datasets: [{
            data: [17.01, 17.11, 17.09, 17.09, 17.05, 17.05, 17.05, 17.04, 17.02, 17.02],
            borderWidth: 4,
            fill: false,
            backgroundColor: "#1F77B4",
            borderColor: "#1F77B4",
            label: "spatial coordinate scale s_x"
        }]
    },
    options: {
        legend: {
            display: false
        },
        scales: {
            y: {
                title: {
                    display: true,
                    text: "SER",
                }
            },
            x: {
                title: {
                    display: true,
                    text:  'spatial coordinate scales',
                }
            },
        },
        elements: {
            line: {
                tension: 0
            }
        }
    }
}

module.exports = chartdata