uniform vec2 size;
uniform int frame;
uniform sampler2D colors;

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
    if(circle &&  rad>1.+fuzz){
        gl_FragColor = bgcol;
        return;
    }

    vec4 c = texture2D(colors, uv);

    //gl_FragColor = vec4((gl_FragColor.rgb*3.+gl_FragColor.gba)/4.,1.);
    c = log(.7+exp(1.*c));
    c /= 1.+abs(c);

    //gl_FragColor = vec4(gl_FragColor[frame/6%4]);
    if(strobe_frames > 0){
        int s = frame/strobe_frames;
        c = vec4(c[int(mod(s,4))], c[int(mod(s+1,4))], c[int(mod(s+2,4))], 0.);

        float sm = float(mod(frame,strobe_frames))/float(strobe_frames);
        c.a = mix(c.r,c.g,sm);
    }

    //gl_FragColor = vec4(c.r, c.g/3., c.g, 1.);
	//gl_FragColor = vec4(c.r);
    //gl_FragColor = vec4(c.a);
   	//gl_FragColor = vec4(c1,c2/4.,c3+c2/2.,1.);
	gl_FragColor = vec4(lchToRgb(hsl_mix2(
        vec3(0.,0.,180.*uv.y), vec3(100.,90.,500.+40.*uv.y), clamp(c.r,0.,1.)
    )),1.);


    //gl_FragColor = gl_FragColor*0.5+0.5;

    if(circle && rad > 1.)
	    gl_FragColor = mix(bgcol, gl_FragColor, vec4(max(0.,1.-(rad-1.)/fuzz)));

	gl_FragColor = clamp(gl_FragColor, 0., 1.);
}
