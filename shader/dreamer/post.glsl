uniform vec2 size;
uniform int frame;
uniform sampler2D color;

const int strobe_frames = 8;

out vec4 fragColor;

const vec2 translate = vec2(-.5);
vec2 get_scale(vec2 res){
    return 2./circle_size*res/min(res.x, res.y);
}
vec2 tex2circ(vec2 x){
    vec2 scale = get_scale(size);
    return car2pol((x+translate)*scale);
}
vec2 circ2tex(vec2 x){
    vec2 scale = get_scale(size);
	return pol2car(x)/scale-translate;
}
vec2 to_center(vec2 x){
 	//get vector to circle center in screen coordinates
    vec2 center = size*circ2tex(vec2(0.));
    return center-x;
}
vec2 wrap(vec2 x){
    vec2 u = tex2circ(x);
    if(u.r>1.){
        u.r -= floor(u.r)+1.;
        return circ2tex(u);
    }
    return x;
}

void main()
{
    vec3 d = vec3(1./size, 0.);
    vec2 uv = gl_FragCoord.xy * d.xy;
    float rad = tex2circ(uv).r;
    if(circle && rad>1.+fuzz){
        fragColor = bgcol;
        return;
    }

    vec4 c = texture(color, uv);

    //fragColor = vec4((fragColor.rgb*3.+fragColor.gba)/4.,1.);
    c = log(.7+exp(1.*c));
    c /= 1.+abs(c);

    //fragColor = vec4(fragColor[frame/6%4]);
    vec4 c_strobe;
    if(strobe_frames > 0){
        int s = frame/strobe_frames;
        c_strobe = vec4(c[s%4], c[(s+1)%4], c[(s+2)%4], 0.);

        float sm = float(frame%strobe_frames)/float(strobe_frames);
        c_strobe.a = mix(c_strobe.r, c_strobe.g, sm);
    }

    fragColor = vec4(lchToRgb(hsl_mix2(
        vec3(0.+10.*clamp(c.a,0.,1.), 0., 160.*uv.y+90.*c.b),
        vec3(100., 90., 480.+40.*uv.y+90.*c.g),
        clamp(c_strobe.a, 0., 1.)
    )),1.);

    //fragColor = fragColor*0.5+0.5;

    if(circle && rad > 1.)
        fragColor = mix(bgcol, fragColor, vec4(max(0.,1.-(rad-1.)/fuzz)));

    fragColor = clamp(fragColor, 0., 1.);
    // fragColor = vec4(test.rgb, 1.);
}
