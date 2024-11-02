window.getWindowDimensions = () => {
    return {
        width: window.innerWidth,
        height: window.innerHeight
    };
};

window.addResizeListener = (dotNetObj) => {
    window.addEventListener('resize', () => {
        dotNetObj.invokeMethodAsync('OnResize');
    });
};

window.removeResizeListener = () => {
    window.removeEventListener('resize');
};
