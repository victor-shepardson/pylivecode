uniform vec2 size;
uniform int frame;
uniform sampler2D history_0;
uniform sampler2D displacements;

out vec4 fragColor;

// fragment colors

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
    if (!circle) return fract(x);
    vec2 u = tex2circ(x);
    if(u.r>1.){
        u.r -= floor(u.r)+1.;
        return circ2tex(u);
    }
    return x;
}

vec4 conv(vec2 uv){
    vec3 d = vec3(1./size, 0.);
    return 0.25*(
        texture(history_0, wrap(uv+d.xz))
        + texture(history_0, wrap(uv-d.xz))
        + texture(history_0, wrap(uv+d.zy))
        + texture(history_0, wrap(uv-d.zy))
	);
}

void main()
{
    vec3 d = vec3(1./size, 0.);
    //vec2 uv = gl_FragCoord.xy * d.xy;
    vec2 uv = (gl_FragCoord.xy + to_center(gl_FragCoord.xy)*zoom + drift) * d.xy;
    float rad = tex2circ(uv).r;
    if(circle && rad>1.+fuzz){
        fragColor = bgcol;
        return;
    }

    vec4 r = texture(displacements, wrap(uv));
    vec2 r1 = r.xy;
    vec2 r2 = r.zw;
    vec4 c0 = texture(history_0, wrap(uv));
    vec4 c1 = texture(history_0, wrap(uv+r1*d.xy));
    vec4 c2 = texture(history_0, wrap(uv+r2*d.xy));

    vec4 dJdc = -term(c0-c1) + term(c0-c2) + lambda_c*c0;

    dJdc -= lambda_b*(conv(uv)-c0);
    //dJdc -= term(c0 - conv(uv));

    fragColor = c0 - alpha_c*dJdc;
	// rad = 1.;
    // if(iFrame==0 || iMouse.z>0.){ //initial condition
    if(frame==0){
        fragColor = cos(pi*rad+vec4(0., 1./4., 1./2., 3./4.));
        //fragColor = vec4(cos(pi*rad+vec3(0., 2./3., 4./3.)),0.);
        //fragColor = vec4(cos(pi*2.*(uv.xxy+vec3(0.,0.5,0.))).gbr, 0.);
    	//fragColor = vec4(sin(pi*2.*vec3(1.,6.,7.)*(uv.yxx+vec3(0.,0.,0.25))).gbr, 0.);
        //fragColor = texture(iChannel2, uv)*2.-1.;
    }
	// fragColor = uv.xxyy;
}
