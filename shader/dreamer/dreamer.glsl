uniform vec2 size;
uniform int frame;
uniform sampler2D history_t0_b0; //colors
uniform sampler2D history_t0_b1; //displacements

const float lambda_b = 0.8;//0.0625;//8e-1;
const float zoom = 0.0;//0.001;//
const vec2 drift = vec2(0.,0.);

const float lambda_c = 3e-2;//5e-2;//2e-1;
const float alpha_c = 0.5;//0.0625;

const float lambda_r = 0.;//1e-3;
const float alpha_r = 0.25;
const float knee = 0.5;

layout(location = 0) out vec4 fragColor;
layout(location = 1) out vec4 fragDisp;

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
        texture(history_t0_b0, wrap(uv+d.xz))
        - texture(history_t0_b0, wrap(uv-d.xz)),
        texture(history_t0_b0, wrap(uv+d.zy))
        - texture(history_t0_b0, wrap(uv-d.zy))
	);
}

vec4 conv(vec2 uv, sampler2D tex){
    vec3 d = vec3(1./size, 0.);
    return 0.25*(
        texture(tex, wrap(uv+d.xz))
        + texture(tex, wrap(uv-d.xz))
        + texture(tex, wrap(uv+d.zy))
        + texture(tex, wrap(uv-d.zy))
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
        fragDisp = vec4(0.);
        return;
    }

    vec4 r = texture(history_t0_b1, wrap(uv));
    vec2 r1 = r.xy;
    vec2 r2 = r.zw;
    vec4 c0 = texture(history_t0_b0, wrap(uv));
    vec4 c1 = texture(history_t0_b0, wrap(uv+r1*d.xy));
    vec4 c2 = texture(history_t0_b0, wrap(uv+r2*d.xy));
    mat4x2 dc1dr1 = grad(uv+r1*d.xy);
    mat4x2 dc2dr2 = grad(uv+r2*d.xy);

    vec4 dJdc1 = term(c1-c0) - term(c1-c2);
    vec4 dJdc2 = -term(c2-c0) + term(c2-c1);

    vec2 dJdr1 = dc1dr1 * dJdc1;
    vec2 dJdr2 = dc2dr2 * dJdc2;

    if(knee > 0.){
        dJdr1 /= length(dJdr1) + knee;
        dJdr2 /= length(dJdr2) + knee;
    }

    vec4 dJdr = vec4(dJdr1, dJdr2);
    dJdr += lambda_r*dJdr;
    dJdr -= lambda_b*(conv(uv, history_t0_b1)-r);

    vec4 dJdc = -term(c0-c1) + term(c0-c2) + lambda_c*c0;
    dJdc -= lambda_b*(conv(uv, history_t0_b0)-c0);

    fragColor = c0 - alpha_c*dJdc;
    fragDisp= r - alpha_r*dJdr;

    if(frame==0){
        if(circle){
            fragColor = cos(pi*rad+vec4(0., 1./4., 1./2., 3./4.));
            // fragDisp = (2.*size/5.).xyxy*vec4(1., 0., 0., -1.);
            // fragDisp = (size/2. - gl_FragCoord.xy).xyxy*vec2(2./3.,4./3.).xxyy;
            vec2 dc = (size/2.-gl_FragCoord.xy);
            // fragDisp = dc.xyxy*vec4(1.,1.,1.,-1.)/(length(dc)+0.1)*min(size.x, size.y)*circle_size/3.;
            fragDisp = dc.xyxy*vec2(2./3.,4./3.).xxyy/(length(dc)+0.1)*min(size.x, size.y)*circle_size/3.;
        } else {
            fragColor = cos(uv.xxyy*vec4(1.,2.,3.,1.)*pi*2.);
            fragDisp = (2*min(size.x, size.y)/3+1)*vec4(1.,0.,0.,1.);
        }
    }
}
