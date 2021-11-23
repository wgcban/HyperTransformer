function xrgb = get_rgb(x)
    R=61;
    G=35;
    B=10;

    xr = x(:,:,R);
    xg = x(:,:,G);
    xb = x(:,:,B);
    xrgb = cat(3, xr, xg, xb)/max(x, [], "all");
end

