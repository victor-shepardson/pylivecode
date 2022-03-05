in vec2 uv;
uniform vec2 size;
uniform vec2 src_size;
uniform int frame;

uniform sampler2D history_t0_b0;
uniform sampler2D src;

out vec4 fragColor;

void main() {
  ivec2 isize = ivec2(size);
  ivec2 p = (ivec2(uv*size) + ivec2(size.x - 1, 0)) % isize;
//   ivec2 p = (ivec2(uv*size) + ivec2(0, size.y - 1)) % ivec2(size);
//   ivec2 p = ivec2(0,0);
//   p = ivec2(0, int(uv.y*size.y));
  vec4 below = texelFetch(history_t0_b0, (p+isize-ivec2(0,1))%isize, 0);
  vec4 above = texelFetch(history_t0_b0, (p+ivec2(0,1))%isize, 0);

  vec4 last = texelFetch(history_t0_b0, p, 0);
  vec2 pos = last.xy;
  vec2 mom = last.zw;

  vec4 color = texture(src, pos);

  vec2 repulse = normalize(
      fract(pos - below.xy + 0.5)
    - fract(pos - above.xy + 0.5) //attract
    - 1)/src_size;

//   vec2 c2 = 2.*vec2(
    //   dot(color.rgb, normalize(vec3(2,-1,-1))), 
    //   dot(color.rgb, normalize(vec3(p.y*1e-4,1,-1))));
  vec2 c2 = normalize( 
    color.rg - color.ba + sin(vec2(0, 1.65)+p.y/size.y)*1e-2
    ) / src_size;

  vec2 new_mom = 0.5 * mom + c2 + repulse;

//   vec2 new_pos = pos + normalize(new_mom*size)/size/2 + 2 ;
  vec2 new_pos = pos + new_mom/5 + 2 ;//+ sin(


  new_pos -= floor(new_pos);

  fragColor = vec4(new_pos, new_mom);
    // fragColor = last;
    // fragColor = vec4(float(p.x!=510));
    // fragColor = vec4(0);
    // fragColor = fract(last + 0.02);
}