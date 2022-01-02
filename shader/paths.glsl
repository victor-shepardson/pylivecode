in vec2 uv;
uniform vec2 size;
uniform vec2 src_size;

uniform sampler2D history_t0_b0;
uniform sampler2D src;

out vec4 fragColor;

void main() {
  ivec2 p = (ivec2(uv*size) + ivec2(size.x - 1, 0)) % ivec2(size);
//   ivec2 p = (ivec2(uv*size) + ivec2(0, size.y - 1)) % ivec2(size);
//   ivec2 p = ivec2(0,0);
//   p = ivec2(0, int(uv.y*size.y));

  vec4 last = texelFetch(history_t0_b0, p, 0);
  vec2 pos = last.xy;

  vec4 color = texture(src, pos);

  vec2 c2 = 2.*vec2(
      dot(color.rgb, normalize(vec3(2,-1,-1))), 
      dot(color.rgb, normalize(vec3(0,1,-1))));

  vec2 new_pos = pos + src_size + c2 + sin(
      color.xy - color.zx * 3.14 * 5 + p.y*0.01);
      
  new_pos -= floor(new_pos/src_size)*src_size;

  fragColor = vec4(new_pos.x, new_pos.y, 1., 1.);
    // fragColor = last;
    // fragColor = vec4(float(p.x!=510));
    // fragColor = vec4(0);
    // fragColor = fract(last + 0.02);
}