function xrgb = get_rgb(x)
    R=60;
    G=30;
    B=10;

    xr = x(:,:,R);
    xg = x(:,:,G);
    xb = x(:,:,B);
    xrgb = cat(3, xr, xg, xb)/max(x, [], "all");
end

