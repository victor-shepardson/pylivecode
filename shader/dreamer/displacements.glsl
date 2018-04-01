uniform vec2 size;
uniform int frame;
uniform sampler2D history_0;
uniform sampler2D colors;

out vec4 fragColor;

//Displacements to neighbors

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

mat4x2 grad(vec2 uv){
    vec3 d = vec3(1./size, 0.);
    return mat4x2(
        texture(colors, wrap(uv+d.xz))
        - texture(colors, wrap(uv-d.xz)),
        texture(colors, wrap(uv+d.zy))
        - texture(colors, wrap(uv-d.zy))
	);
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

    vec4 r = texture(history_0, wrap(uv));
    vec2 r1 = r.xy;
    vec2 r2 = r.zw;
    vec4 c0 = texture(colors, wrap(uv));
    vec4 c1 = texture(colors, wrap(uv+r1*d.xy));
    vec4 c2 = texture(colors, wrap(uv+r2*d.xy));
    mat4x2 dc1dr1 = grad(uv+r1*d.xy);//mat3x2(dFdx(c1), dFdy(c1));
    mat4x2 dc2dr2 = grad(uv+r2*d.xy);//mat3x2(dFdx(c2), dFdy(c2));

    vec4 delta_10 = c1-c0;
    vec4 delta_12 = c1-c2;
    vec4 delta_20 = c2-c0;
    vec4 delta_21 = -delta_12;

    vec4 dJdc1 = term(c1-c0) - term(c1-c2);
    vec4 dJdc2 = -term(c2-c0) + term(c2-c1);

    vec2 dJdr1 = dc1dr1 * dJdc1;
    vec2 dJdr2 = dc2dr2 * dJdc2;

	if(knee > 0.){
	    dJdr1 /= length(dJdr1) + knee;
	    dJdr2 /= length(dJdr2) + knee;
	}

    vec4 dJdr = vec4(dJdr1, dJdr2) ;

    dJdr += lambda_r*dJdr;
    dJdr -= lambda_b*(conv(uv)-r);
    //dJdr -= term(r - conv(uv));

    fragColor = r - alpha_r*dJdr;
    //fragColor = (fract(fragColor*d.xyxy+0.5)-0.5)*sizexy;

    if(frame==0){
        /*fragColor = (
            normalize(to_center(gl_FragCoord.xy)).xyxy
            *vec4(2.,2.,-1.,-1.).xxyy
            /max(d.x,d.y)*circle_size/8.
            );*/
        /*fragColor = (
            vec4(normalize(to_center(gl_FragCoord.xy)), vec2(0.,1.))
            /max(d.x,d.y)*circle_size/3.
            );*/
		//fragColor = min(size.x,size.y)*vec4(0., 1., 1., 0.)*sqrt(2.)/4.;
        //fragColor = (size/3.-1.).xyxy*vec4(1., 1., 1., -1.);
        fragColor = (size/4.).xyxy*vec4(1., 0., 0., -1.);
        //fragColor = vec4(1.,1.,-1.,1.);
        //vec2 m = pow(vec2(2.), 4.*sin(pi*2.*(uv.y+vec2(0.5,0.75))));
    	//fragColor = m.xxyy*sizexy*0.03125*sin(2.*pi*(uv.x+vec2(0.5,0.25))).xyyx*vec4(1.,1.,1.,-1.);
        //fragColor = sizexy/16.*(2.+sin(pi*2.*(uv.x+vec4(0.,1.,2.,3.)/4.)))*vec4(1., 1., 1., -1.);
    	//fragColor = 128.*vec4(1., 1., 1., -1.);
    }
}
