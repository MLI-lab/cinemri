const chartdata = {
    type: 'line',
    data: {
        labels: [1, 2.5, 5.3, 7, 10, 20],
        datasets: [{
            data: [17.05, 17.05, 17.05, 17.04, 17.00, 16.94],
            borderWidth: 4,
            fill: false,
            backgroundColor: "#1F77B4",
            borderColor: "#1F77B4",
            label: "temporal coordinate scale"
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
                    text:  'temporal coordinate scale',
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